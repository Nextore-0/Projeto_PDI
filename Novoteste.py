"""
Sistema de Análise de Cirurgia de Catarata com Visão Computacional
Implementa técnicas de PDI clássicas + Deep Learning para rastreamento e análise
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas opcionais para Deep Learning
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch não disponível. Usando apenas técnicas clássicas.")


# =====================================================================
# ------------------------ KALMAN FILTER ------------------------------
# =====================================================================

class KalmanTracker:
    """Filtro de Kalman para rastreamento"""

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, measurement):
        measurement = np.array([[np.float32(measurement[0])],
                                [np.float32(measurement[1])]])
        self.kalman.correct(measurement)
        return self.kalman.predict()


# =====================================================================
# -------------------- IRIS / PUPIL DETECTOR --------------------------
# =====================================================================

class IrisPupilDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.kalman = KalmanTracker()

    def detect_with_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        iris_idx = [469, 470, 471, 472]
        pts = []
        for i in iris_idx:
            lm = landmarks.landmark[i]
            pts.append([int(lm.x * w), int(lm.y * h)])

        center = np.mean(pts, axis=0).astype(int)
        radius = int(dist.euclidean(pts[0], pts[2]) / 2)

        return center, radius, pts

    def track(self, frame):
        center, radius, pts = self.detect_with_mediapipe(frame)
        if center is None:
            return None, None

        kpred = self.kalman.update(center)
        return kpred[:2].flatten().astype(int), radius


# =====================================================================
# -------------------- CAPSULORHEXSIS ANALYZER ------------------------
# =====================================================================

class CapsulorrhexisAnalyzer:

    def analyze_circularity(self, contour):
        if len(contour) < 5:
            return 0.0

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0.0

        circ = 4 * np.pi * area / (perimeter ** 2)
        return min(circ * 100, 100)

    def detect_capsulorrhexis(self, frame, center, radius):

        x, y = center
        r = radius + 20

        y1, y2 = max(0, y - r), min(frame.shape[0], y + r)
        x1, x2 = max(0, x - r), min(frame.shape[1], x + r)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        cnt = max(contours, key=cv2.contourArea)
        quality = self.analyze_circularity(cnt)

        cnt[:, 0, 0] += x1
        cnt[:, 0, 1] += y1

        return cnt, quality


# =====================================================================
# -------------------- OPTICAL FLOW TRACKER ---------------------------
# =====================================================================

class OpticalFlowTracker:

    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None

    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, 100, 0.3, 7)
            return None

        pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(15, 15), maxLevel=2
        )

        if pts is None:
            return None

        good_new = pts[status == 1]
        good_old = self.prev_pts[status == 1]

        self.prev_gray = gray.copy()
        self.prev_pts = good_new.reshape(-1, 1, 2)

        return good_new, good_old


# =====================================================================
# ---------------------- PRINCIPAL SYSTEM ------------------------------
# =====================================================================

class CataractSurgeryAnalyzer:

    def __init__(self):
        self.iris = IrisPupilDetector()
        self.caps = CapsulorrhexisAnalyzer()
        self.flow = OpticalFlowTracker()

        self.metrics = {"capsulorrhexis_quality": []}

    def process_frame(self, frame):

        results = {}

        # IRIS TRACKING
        center, radius = self.iris.track(frame)
        if center is not None:
            cv2.circle(frame, tuple(center), radius, (0, 255, 0), 2)
            cv2.circle(frame, tuple(center), 4, (0, 0, 255), -1)
            results["iris_center"] = center
            results["iris_radius"] = radius

            contour, quality = self.caps.detect_capsulorrhexis(frame, center, radius)
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
                results["capsulorrhexis_quality"] = quality
                self.metrics["capsulorrhexis_quality"].append(quality)

        # ======================
        # OPTICAL FLOW CORRIGIDO
        # ======================
        flow = self.flow.track(frame)
        if flow is not None:
            new, old = flow
            for n, o in zip(new, old):

                a, b = n.ravel()
                c, d = o.ravel()

                # Ignorar pontos inválidos
                if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d):
                    continue

                a, b, c, d = int(a), int(b), int(c), int(d)

                # Garantir que está dentro da tela
                if 0 <= a < frame.shape[1] and 0 <= b < frame.shape[0] and \
                   0 <= c < frame.shape[1] and 0 <= d < frame.shape[0]:

                    cv2.line(frame, (a, b), (c, d), (0, 255, 255), 2)
                    cv2.circle(frame, (a, b), 3, (255, 0, 0), -1)

        return frame, results

    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("ERRO: Não foi possível abrir o vídeo.")
            return

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original = frame.copy()
            processed, results = self.process_frame(frame)

            # ---------- PAINEL DE MÉTRICAS ----------
            metrics = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(metrics, "METRICAS EM TEMPO REAL", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y = 60

            if "iris_center" in results:
                cx, cy = results["iris_center"]
                cv2.putText(metrics, f"Iris: ({cx},{cy})", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 30

            if "iris_radius" in results:
                cv2.putText(metrics, f"Raio: {results['iris_radius']}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 30

            if "capsulorrhexis_quality" in results:
                cv2.putText(metrics, f"Capsulorrexe: {results['capsulorrhexis_quality']:.1f}%",
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # ---------- EXIBIR AS JANELAS ----------
            cv2.imshow("Video Original", original)
            cv2.imshow("Video Analisado", processed)
            cv2.imshow("Metricas", metrics)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()


# =====================================================================
# ------------------------- MAIN --------------------------------------
# =====================================================================

def main():

    print("\nIniciando sistema...")
    import time; time.sleep(2)

    analyzer = CataractSurgeryAnalyzer()

    video_path = "cirurgia.mp4"

    print(f"Processando vídeo: {video_path}")
    analyzer.process_video(video_path)


if __name__ == "__main__":
    main()
