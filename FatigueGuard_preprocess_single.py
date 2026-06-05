import scipy.optimize as opt
import cv2
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gaze_tracking.gui_opencv as gcv
from utilities.kalman import Kalman
from sfm.sfm_module import SFM
import utilities.utils as util
from l2cs import Pipeline, select_device
from gaze_tracking.model import EyeModel
from pathlib import Path
import argparse

def load_matrix(path: str) -> np.ndarray:
    mat = np.load(path)
    mat = np.asarray(mat, dtype=np.float64)
    # if mat.shape != (4, 4):
    #     raise ValueError(f"矩阵形状必须是 (4,4)，但读取到 {mat.shape}: {path}")
    return mat

def pitch_yaw_to_gaze_vector(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """和旧版保持一致：pitch/yaw(弧度) -> 3D单位注视向量(3,)。"""
    gaze_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
    gaze_y = np.sin(pitch_rad)
    gaze_z = np.cos(yaw_rad) * np.cos(pitch_rad)
    gaze = np.array([gaze_x, gaze_y, gaze_z], dtype=np.float64)
    n = np.linalg.norm(gaze)
    return gaze / n if n > 0 else gaze



def mm_to_pixel(vec_mm: np.ndarray, width_px: int, height_px: int, width_mm: float, height_mm: float) -> np.ndarray:
    vec_mm = np.asarray(vec_mm, dtype=np.float64).reshape(3)
    x_px = int(vec_mm[0] * width_px / width_mm)
    y_px = int(vec_mm[1] * height_px / height_mm)
    z = float(vec_mm[2])
    return np.array([x_px, y_px, z], dtype=np.float64)


def _to_jsonable(value):
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value

class GazeToPoint:
    """
    1. 通过标定得到 STransG（屏幕坐标系下的旋转+平移矩阵），将 gaze 投影到屏幕上。
    2. 通过 SFM 得到 WTransG（世界坐标系下的旋转+平移矩阵），将 gaze 投影到屏幕上。
    """
    def __init__(self, directory, args) -> None:
        self.dir = directory
        self.width, self.height, self.width_mm, self.height_mm = gcv.getScreenSize()
        self.df = pd.DataFrame()
        self.args = args
        self.STransG = load_matrix(self.args.stg_npy)
        self.StG = load_matrix(self.args.stg_aux_npy)
        self.scaleWtG = load_matrix(self.args.scale_wtg).item()
        self.STransW = load_matrix(self.args.stw_npy)
        self.StW = load_matrix(self.args.stw_aux_npy)
        self.QueueGaze = np.nan * np.zeros((3, 5))
        self.sfm = SFM(directory, args)
        self.camera_matrix, self.dist_coeffs = gcv.ReadCameraCalibrationData(args.camera_data_dir)
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)

    @staticmethod
    def _frame_timestamp_seconds(cap, frame_idx: int) -> float:
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms is not None and np.isfinite(timestamp_ms) and timestamp_ms > 0:
            return float(timestamp_ms / 1000.0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and np.isfinite(fps) and fps > 0:
            return float(frame_idx / fps)

        return float(frame_idx)

    @staticmethod
    def _extract_face_features(model, frame, gaze_result):
        bbox = None
        landmarks = None

        try:
            face_boxes = model.face_detection.predict(frame)
        except Exception:
            face_boxes = []

        if face_boxes:
            face_box = face_boxes[0]
            bbox = [float(v) for v in face_box]
            try:
                face = model.get_crop_image(frame, face_box)
                if face is not None and face.size > 0:
                    face_landmarks = model.facial_landmark_35.predict(face)
                    xmin, ymin, _, _ = face_box
                    landmarks = [[float(point[0] + xmin), float(point[1] + ymin)] for point in face_landmarks]
            except Exception:
                landmarks = None

        if bbox is None and getattr(gaze_result, "bboxes", None) is not None and len(gaze_result.bboxes):
            bbox = [float(v) for v in np.asarray(gaze_result.bboxes[0]).reshape(-1).tolist()]

        if landmarks is None and getattr(gaze_result, "landmarks", None) is not None and len(gaze_result.landmarks):
            landmarks = _to_jsonable(np.asarray(gaze_result.landmarks[0]))

        return bbox, landmarks

    def RunGazeOnScreen(self, model, cap, sfm=False):
        """ Present different trajectories on screen and record gaze
        """

        if cap != None:
            out_video, wc_width, wc_height = gcv.get_out_video(cap, os.path.join(self.dir, "results"), file_name = "output_video.mp4", scalewidth=2)

        # white_frame = gcv.getWhiteFrame(self.width, self.height)
        # target = gcv.Targets(self.width, self.height)
        frame_prev = None
        WTransG1 = np.eye(4)
        # target.setSetPos([int(self.width/8), int(self.height/8)])   # for DrawSpecificTarget()
        FSgaze = np.array([[-10],[-10],[-10]])
        frame_idx = 0
        weights_path = Path(self.args.weights)
        gaze_pipeline = Pipeline(weights=weights_path, arch=self.args.arch, device=select_device(self.args.device, batch_size=1))
        jsonl_path = Path(self.args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonl_path.open("w", encoding="utf-8") as jsonl_fp:
            while cap.isOpened():
                
                # gazeframe, SetPos = target.DrawTargetGaze(white_frame, self._mm2pixel(FSgaze))
                # gazeframe, SetPos = target.DrawRectangularTargets(white_frame, self._mm2pixel(FSgaze))
                # gazeframe, SetPos = target.DrawSingleTargets(white_frame, self._mm2pixel(FSgaze))
                # gazeframe, SetPos = target.DrawTargetInMiddle(white_frame, self._mm2pixel(FSgaze))

                try:
                    ret, frame = cap.read()
                except StopIteration:
                    break
                if not ret or frame is None:
                    print("Video stream ended")
                    break

                # gray_image, prediction, morphedMask, falseColor, centroid = model.get_iris_Cnn(frame)
                # Undistort the image
                # frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                gaze_result = gaze_pipeline.step(frame)

                pitch = float(gaze_result.pitch[0]) if getattr(gaze_result, "pitch", None) is not None and len(gaze_result.pitch) else np.nan
                yaw = float(gaze_result.yaw[0]) if getattr(gaze_result, "yaw", None) is not None and len(gaze_result.yaw) else np.nan
                confidence = float(gaze_result.scores[0]) if getattr(gaze_result, "scores", None) is not None and len(gaze_result.scores) else np.nan

                bbox, landmarks = self._extract_face_features(model, frame, gaze_result)

                if np.isfinite(pitch) and np.isfinite(yaw):
                    gaze = pitch_yaw_to_gaze_vector(pitch, yaw)
                    gaze = util.MedianFilter(self.QueueGaze, gaze)
                    # with open("./debug_gaze_util_MedianFilter.txt", "a") as f:
                    #     f.write(f"{frame_idx}, {gaze[0]}, {gaze[1]}, {gaze[2]}\n")

                    if frame_prev is not None and sfm:
                        WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)        # WtG1 is a unit vector, has to be scaled   

                    frame_prev = frame
                    """
                    FSgaze	最终融合后的 gaze 屏幕点
                    Sgaze	全局矩阵预测
                    Sgaze2	局部矩阵预测
                    """
                    if sfm:

                        FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
                    else:
                        FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)
                else:
                    gaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                    FSgaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                    Sgaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                    Sgaze2 = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

                s_point_px = mm_to_pixel(FSgaze, self.width, self.height, self.width_mm, self.height_mm)
                x_px = int(np.clip(s_point_px[0], 0, self.width - 1)) if np.isfinite(s_point_px[0]) else -1
                y_px = int(np.clip(s_point_px[1], 0, self.height - 1)) if np.isfinite(s_point_px[1]) else -1

                timestamp = self._frame_timestamp_seconds(cap, frame_idx)
                record = {
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                    "pitch_yaw_rad": [pitch, yaw],
                    "gaze_xyz": [
                        float(gaze[0]) if np.isfinite(gaze[0]) else np.nan,
                        float(gaze[1]) if np.isfinite(gaze[1]) else np.nan,
                        float(gaze[2]) if np.isfinite(gaze[2]) else np.nan,
                    ],
                    "gaze_screen_xy_mm": [
                        float(FSgaze[0]) if np.isfinite(FSgaze[0]) else np.nan,
                        float(FSgaze[1]) if np.isfinite(FSgaze[1]) else np.nan,
                    ],
                    "gaze_screen_xy_px": [x_px, y_px],
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "confidence": confidence,
                }
                if frame_idx != 0:  # 只有不是第一帧时才写入文件
                    jsonl_fp.write(json.dumps(_to_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
                    jsonl_fp.flush()

                frame_idx += 1

            # if out_video is not None:
            #     final_frame = np.concatenate((cv2.flip(cv2.resize(gazeframe, (wc_width, wc_height)), 1), frame), axis=1)
            #     out_video.write(final_frame)
            #     # out_video.write(frame)

            # key_pressed = cv2.waitKey(60)
            # if key_pressed == 27:
            #     break
        cap.release()
        # out_video.release()
        cv2.destroyAllWindows()
        return

    def calibrate(self, model, cap, sfm=False):
        frame = gcv.getWhiteFrame(self.width, self.height)
        if cap != None:
            out_video,_,_ = gcv.get_out_video(cap, os.path.join(self.dir, "results"))
            self.WC_width, self.WC_height = gcv.getWebcamSize(cap)

        target = gcv.Targets(self.width, self.height)
        frame_prev = None
        WTransG1 = np.zeros((4,4))
        target.tstart = time.time()
        while (cap.isOpened()):
            idx, SetPos = target.getTargetCalibration()
            if idx == None:
                break
            
            """ Draw Target on white frame """
            frame2 = frame.copy()
            cv2.circle(frame2, tuple(SetPos), 15, (0, 0, 255), -1)
            length = 25
            thickness = 4
            if idx == 9:
                cv2.arrowedLine(frame2, (SetPos[0]+length, SetPos[1]), (SetPos[0]-length, SetPos[1]), (0, 0, 0), thickness)
            if idx == 10:                
                cv2.arrowedLine(frame2, (SetPos[0]-length, SetPos[1]), (SetPos[0]+length, SetPos[1]), (0, 0, 0), thickness)
            if idx == 11:
                cv2.arrowedLine(frame2, (SetPos[0], SetPos[1]-length), (SetPos[0], SetPos[1]+length), (0, 0, 0), thickness)
            if idx == 12:
                cv2.arrowedLine(frame2, (SetPos[0], SetPos[1]+length), (SetPos[0], SetPos[1]-length), (0, 0, 0), thickness)

            gcv.display_window(frame2)

            try:
                ret, frame_cam = cap.read()
            except Exception as e:
                print(f"Could not read from video stream: {e}")
            if ret == False:
                print("Video stream ended")
                break
            
            if frame_prev is not None and sfm:
                WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame_cam)

            frame_prev = frame_cam.copy()

            # frame_cam = cv2.undistort(frame_cam, self.camera_matrix, self.dist_coeffs)
            eye_info = model.get_gaze(frame=frame_cam, imshow=False)
            if eye_info is None:
                raise Exception("No eye info. Eye tracking failed.")
            
            arr = np.array([])
            for i in pd.Series(eye_info).values:
                arr = np.hstack((arr,i))
            timestamp = time.time_ns()/1000000000
            SetPos = self._pixel2mm(SetPos)
            self.df = pd.concat([ self.df, pd.DataFrame([np.hstack((timestamp, idx, arr, SetPos, 0, WTransG1.flatten()))]) ])

            if out_video is not None:      
                out_video.write(frame_cam)
            else:
                print("No output video")

            key_pressed = cv2.waitKey(1)   # this is needed
            if key_pressed == 27:
                exit()

        cv2.destroyAllWindows()
        out_video.release()

        self.df.columns = ['Timestamp', 'idx', 'gaze_x', 'gaze_y', 'gaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadBox_xmin', 'HeadBox_ymin', 'RightEyeBox_xmin', 'RightEyeBox_ymin', 'LeftEyeBox_xmin', 'LeftEyeBox_ymin', 'ROpenClose','LOpenClose', 'set_x', 'set_y', 'set_z'] + 16*['WTransG']
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.dir, "results", "Calibration.csv"))

        gaze, SetVal, WTransG, g = self._RemoveOutliers()
    
        if sfm:
            STransW, scaleWtG, STransG = self._fitSTransG_sfm(gaze, SetVal, WTransG, g)
        else:
            STransG = self._fitSTransG(gaze, SetVal, g)

        Sg, SgCalib = self._getCalibValuesOnScreen(g, STransG)
        """ Plot Gaze On Screen"""
        self._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")
        self._WriteStatsInFile(STransG)

        return STransG

    def _getGazeOnScreen(self, gaze):
        scaleGaze = self._getScale(gaze, self.STransG)
        Sgaze = (self.STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """   
        for i in range(len(self.StG)):
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)

        """
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system
        Sgaze2 = gaze vector from calibration point
        """
        return FSgaze, Sgaze, Sgaze2

    def _getGazeOnScreen_sfm(self, gaze, WTransG):
        WTransG[:3,3] = self.scaleWtG*WTransG[:3,3]
        STransG = self.STransW @ WTransG
        scaleGaze = self._getScale(gaze, STransG)
        Sgaze = (STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """   
        for i in range(len(self.StW)):
            STransG_ = np.vstack((np.hstack((SRotW, self.StW[i].reshape(3,1))), np.array([0,0,0,1]))) @ WTransG
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)
        """
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system with head movement
        Sgaze2 = gaze vector from calibration point with head movements
        """
        return FSgaze, Sgaze, Sgaze2

    def _fitSTransG(self, gaze, SetVal, g):
        
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 

        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        gaze = gaze[:,:,None]

        """ Without sfm """
        def alignError(x, *const):
            SRotG, gaze, SetVal = const
            StG = np.array([[x[0]],[x[1]],[x[2]]])
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-SRotG.T @ StG))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + StG
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()
        
        const = (SRotG, gaze, SetVal)
        x0 = np.array([self.width/2, self.height/2, self.width])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StG = np.array([[xopt[0]],[xopt[1]],[xopt[2]]])
        STransG = np.r_[np.c_[SRotG, StG], np.array([[0,0,0,1]])]

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]

        self.STransG = STransG

        return STransG
    
    def _fitSTransG_sfm(self, gaze, SetVal, WTransG, g):
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 
        WTransG = WTransG.to_numpy().reshape(-1,4,4)

        WRotG = WTransG[:,:3,:3]
        WtG = WTransG[:,:3,3]
        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

        gaze = gaze[:,:,None]

        """ Model over camera coordinate system getting gaze from SFM  """
        def alignError(x, *const):
            SRotW, WRotG, gaze, WtG, SetVal = const
            StW = np.array([[x[1]],[x[2]],[0]])
            SRotG = SRotW @ WRotG
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-np.transpose(SRotG, axes=(0,2,1)) @ (SRotW @ (x[0]*WtG[:,:,None]) + StW)))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + SRotW @  (x[0]*WtG[:,:,None]) + StW
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()

        const = (SRotW, WRotG, gaze, WtG, SetVal)
        x0 = np.array([1, self.width/2, self.height/2])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StW = np.array([[xopt[1]],[xopt[2]],[0]])
        self.STransW = np.r_[np.c_[SRotW, StW], np.array([[0,0,0,1]])]
        WTransG = np.concatenate((np.c_[WRotG, xopt[0]*WtG[:,:,None]], np.tile(np.array([[0, 0, 0, 1]]), (WtG.shape[0], 1, 1))), axis=1)
        STransG = self.STransW @ np.median(WTransG, axis=0)
        self.scaleWtG = xopt[0]

        WtG = np.median(WtG[:,:,None], axis=0)

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StW = [None]*size
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]
            self.StW[i] = STransG_[:3,3,None] - SRotW @ (self.scaleWtG*WtG)

        self.STransG = STransG

        return self.STransW, self.scaleWtG, STransG
        
    def _getCalibValuesOnScreen(self, g, STransG):
        Sg = [None]*len(g)
        SgCalib = [None]*len(g)
        # SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        SRotG = STransG[:3,:3]
        for i in range(len(g)):
            gaze = g[i].to_numpy()
            scaleGaze = self._getScale(gaze, STransG)
            Sg[i] = (STransG @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            SgCalib[i] = (STransG_ @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]

        return Sg, SgCalib

    def _getSTransG(self, SRotG, SposA, gazeVector, scaleGaze):
        STransA = np.vstack((np.hstack((np.eye(3), SposA)), np.array([0,0,0,1])))      
        ATransG = np.vstack((np.hstack((SRotG, -SRotG.T @ (scaleGaze*gazeVector[:,None]))), np.array([0,0,0,1])))
        STransG = STransA @ ATransG
        GTransS = np.vstack((np.hstack((STransG[0:3,0:3].T, -STransG[0:3,0:3].T @ STransG[0:3,3].reshape(3,1))), np.array([0,0,0,1])))

        return STransG, GTransS

    def _getScale(self, gaze, STransG):
        Gz = np.array([[0],[0],[1]])
        GTransS = util.invHomMatrix(STransG)
        GtS = GTransS[:3,3].reshape(3,1)
        if np.ndim(gaze) == 1:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,None])
        elif np.ndim(gaze) == 2:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,:,None])

        return scaleGaze

    def _ProjectVetorOnPlane(self, Trans, vector):
        """ Translation of homogenous Trans-Matrix must be in same coordinate system as Vector """
        vector = vector.reshape(3,1)
        # VectorNormal2Plane = (Trans @ np.array([[0],[0],[1],[1]]))[0:3]
        VectorNormal2Plane = (Trans[:3,:3] @ np.array([[0],[0],[1]]))
        # Gz = self.GTransB[0:3,2].reshape(3,1) # not sure why this would work for Tobii (was implemented before)
        transVec = Trans[:3,3]
        t = (VectorNormal2Plane.T @ transVec) / (VectorNormal2Plane.T @ vector)
        Vector2Plane = np.vstack((t*vector, 1))
        return Vector2Plane

    def _RemoveOutliers(self):
        """ Remove Outliers """
        idx = int(pd.unique(self.df['idx'])[-1])+1  # if head turning use -3 otherwise +1
        g = [None]*idx
        s = [None]*idx
        WTG = [None]*idx
        for i in range(idx):            
            g_ = self.df[self.df['idx'].values==i].loc[:,'gaze_x':'gaze_z']
            # sign = np.sign(np.median(g_, axis=0)[0])
            set_val = self.df[self.df['idx'].values==i].loc[:,'set_x':'set_z']
            WTG_ = self.df[self.df['idx'].values==i].filter(like='WTransG')
            mask = self._MaskOutliers(g_.loc[:,'gaze_x']) & self._MaskOutliers(g_.loc[:,'gaze_y']) #& (sign*g_.loc[:,'gaze_x'] > 0)          
            g[i] = g_[mask]
            s[i] = set_val[mask]
            WTG[i] = WTG_[mask]
        
        self.SetValues = [v.to_numpy()[0][:,None] for v in s]
        gaze = pd.concat(g, axis=0)
        SetVal = pd.concat(s, axis=0)
        W_T_G = pd.concat(WTG, axis=0)

        return gaze, SetVal, W_T_G, g

    def _MaskOutliers(self, arr, std_threshold=1):
        """
        Removes outliers from a NumPy array using the standard deviation method.
        Parameters:
            arr (numpy.ndarray): The input array.
            std_threshold (float): The number of standard deviations from the mean to use as the threshold for outlier detection.
        Returns:
            numpy.ndarray: The mask to remove outliers.
        """
        mean = np.mean(arr)
        std = np.std(arr)
        threshold = std_threshold * std
        mask = np.abs(arr - mean) < threshold
        return mask

    def _MaskOutliersPercentile(self, array):
        q75,q25 = np.percentile(array,[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        return (array > min) & (array < max)

    def _WriteStatsInFile(self, STransG):
        """ Write stats in file """
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        with open(os.path.join(self.dir, "results", 'stats.txt'), 'w') as f:
            f.write(f"Transformation matrices: \n")
            f.write(f"STransG1\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[0].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG2\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[1].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG3\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[2].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG4\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[3].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"Screen Information: \n")
            f.write(f"Width: {self.width}px, {self.width_mm}mm\n")
            f.write(f"Height: {self.height}px, {self.height_mm}mm\n")
            f.write(f"Webcam Information: \n")
            f.write(f"Width: {self.WC_width}px\n")
            f.write(f"Height: {self.WC_height}px\n")

    def _getARotG(self, p_origin, p_xCoord, p_yCoord):
        """ Rotation Matrix """
        GxA = p_xCoord - p_origin
        GxA = GxA/np.linalg.norm(GxA)
        GyA = p_yCoord - p_origin
        GyA = GyA/np.linalg.norm(GyA)
        GzA = self._cross(GxA, GyA)
        GRotA = np.hstack((GxA.reshape(3,1), GyA.reshape(3,1), GzA.reshape(3,1)))
        ARotG = GRotA.transpose()

        return ARotG

    def _mm2pixel(self, vector_mm):
        vector = vector_mm.copy()
        if vector.ndim == 2 and vector.shape[0] == 3:
            vector[0] = int(vector[0] * self.width/self.width_mm)
            vector[1] = int(vector[1] * self.height/self.height_mm)
            vector[2] = int(vector[2])
        elif vector.ndim == 3 and vector.shape[1] == 3:
            vector[:,0] = (vector[:,0] * self.width/self.width_mm).astype(int)
            vector[:,1] = (vector[:,1] * self.height/self.height_mm).astype(int)
            vector[:,2] = (vector[:,2]).astype(int)
        else:
            raise Exception("Vector has wrong shape")

        return vector

    def _pixel2mm(self, vector_px):
        if isinstance(vector_px, list):
            vector_px = np.array(vector_px)
        vector = vector_px.copy()
        if vector.ndim == 1 and vector.shape[0] == 2:
            vector[0] = vector[0] * self.width_mm/self.width
            vector[1] = vector[1] * self.height_mm/self.height
        elif vector.ndim == 2 and vector.shape[1] == 2:
            vector[:,0] = vector[:,0] * self.width_mm/self.width
            vector[:,1] = vector[:,1] * self.height_mm/self.height
        else:
            raise Exception("Vector has wrong shape")

        return vector

    def _PlotGaze2D(self, g, Sg, SgCalib, name="GazeOnScreen"):

        # Sg1 = self._mm2pixel(Sg1)
        # Sg2 = self._mm2pixel(Sg2)
        # Sg3 = self._mm2pixel(Sg3)
        # Sg4 = self._mm2pixel(Sg4)
        # SetBp1 = self._mm2pixel(self.SetValues[0])
        # SetBp2 = self._mm2pixel(self.SetValues[1])
        # SetBp3 = self._mm2pixel(self.SetValues[2])
        # SetBp4 = self._mm2pixel(self.SetValues[3])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

        legend = [None]*len(g)
        for i in range(len(g)):
            """ Axis 0: Raw gaze points """
            gaze = g[i].to_numpy()
            ax[0].scatter(gaze[:,0],gaze[:,1])            
            legend[i] = f"p{i+1} values"
            """ Axis 1: Gaze on screen """
            ax[1].scatter(Sg[i][:,0],Sg[i][:,1])

        for i in range(len(g)):
            gaze = g[i].to_numpy()
            ax[0].plot(np.median(gaze[:,0]),np.median(gaze[:,1]),'r+', linewidth=4,  markersize=12)
            ax[1].plot(np.median(Sg[i][:,0]),np.median(Sg[i][:,1]),'r+', linewidth=4,  markersize=12)
            # ax[1].plot(np.median(SgCalib[i][:,0]),np.median(SgCalib[i][:,1]),'k+', linewidth=4,  markersize=12)
            ax[1].plot(self.SetValues[i][0],self.SetValues[i][1],'y*', linewidth=4, markersize=12)


        # ax[0].legend(legend+["Median gaze point"])
        ax[0].set_title('x-y-corrdinates of raw unit gaze points')
        ax[0].set_xlabel("x-direction (unit length)")
        ax[0].set_ylabel("y-direction (unit length)")
        ax[0].grid()
        # ax[1].legend(legend+["Median gaze point", "Displayed Point"])
        ax[1].set_xlabel("x-direction (mm)")
        ax[1].set_ylabel("y-direction (mm)")
        # ax[1].set_title(f"Gaze on screen with resolution {self.width}x{self.height}")
        ax[1].set_title(f"Gaze on screen with dimensions {self.width_mm}mmx{self.height_mm}mm")
        ax[1].grid()

        plt.savefig(os.path.join(self.dir, "results", name))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2CS gaze -> screen point (minimal)")
    p.add_argument("--device", default="cpu", type=str, help="cpu 或 cuda:0")
    p.add_argument("--weights", default="models/L2CSNet_gaze360.pkl", type=str, help="L2CS 权重文件路径")
    p.add_argument("--arch", default="ResNet50", type=str, help="ResNet18/34/50/101/152")
    p.add_argument("--input", default="/data3/wangchangmiao/shenxy/Code/gaze/selfData/1.mp4"
                   , type=str, help="输入视频路径")

    p.add_argument("--directory", default=".", type=str, help="输出目录（含results等）")

    p.add_argument("--mode", choices=["global", "sfm"], default="sfm", help="global=使用STransG, sfm=使用STransW@WTransG")
    p.add_argument("--stg_npy", default="STransG.npy", help="STransG.npy 路径")
    p.add_argument("--stw_npy", default="STransW.npy", help="STransW.npy 路径（sfm模式需要）")
    p.add_argument("--scale_wtg", default="scaleWtG.npy", type=str, help="scaleWtG.npy 路径（sfm模式需要）")

    p.add_argument("--stg_aux_npy", default="StG.npy", help="辅助点平移 StG.npy（形状: Kx3x1 或 Kx3）")
    p.add_argument("--stw_aux_npy", default="StW.npy", help="辅助点平移 StW.npy（形状: Kx3x1 或 Kx3，sfm模式用）")
    p.add_argument("--sfm_openvino_device", default="CPU", type=str, help="SFM 的 OpenVINO 推理设备（默认CPU）")

    p.add_argument("--max_frames", default=0, type=int, help="最多处理多少帧，0表示全部")
    p.add_argument("--jsonl", default="result/l2cs_screen_points.jsonl", help="输出JSONL路径")
    p.add_argument("--camera_data_dir", default=None, type=str, help="相机标定数据目录路径")
    return p.parse_args()

## 正确代码
if __name__ == '__main__':
    args = parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    gaze_to_point = GazeToPoint(directory, args)
    cap = cv2.VideoCapture(args.input)
    gaze_to_point.RunGazeOnScreen(model= EyeModel(directory), cap=cap, sfm=(args.mode=="sfm"))