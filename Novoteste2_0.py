"""
Sistema de Análise de Cirurgia de Catarata com Visão Computacional
Versão Otimizada com ROI Central e Hough Circles
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
# ---------------------- PRINCIPAL SYSTEM ------------------------------
# =====================================================================

class CataractSurgeryAnalyzer:

    def __init__(self):
        self.iris = IrisPupilDetector()
        self.caps = CapsulorrhexisAnalyzer()
        
        # Variáveis para optical flow
        self.prev_gray = None
        self.prev_points = None
        self.frame_counter = 0

        self.metrics = {"capsulorrhexis_quality": []}

    def adjust_roi_size(self, frame, roi_percentage=0.5):
        """
        Permite ajustar dinamicamente o tamanho da ROI
        roi_percentage: 0.3 a 0.8 (30% a 80% do frame)
        """
        h, w = frame.shape[:2]
        roi_width = int(w * roi_percentage)
        roi_height = int(h * roi_percentage)
        
        x1 = (w - roi_width) // 2
        y1 = (h - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        
        return x1, y1, x2, y2

    def process_frame(self, frame):
        """
        Processa frame com foco na região central para detecção precisa da íris
        """
        h, w = frame.shape[:2]
        
        # =============================
        # 1. DEFINIR ROI CENTRAL (Região de Interesse)
        # =============================
        roi_width = int(w * 0.85)   # 50% da largura
        roi_height = int(h * 0.85)  # 50% da altura
        
        x1 = (w - roi_width) // 2
        y1 = (h - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        
        # Desenha retângulo da ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "ROI Central", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # =============================
        # 2. PRÉ-PROCESSAMENTO OTIMIZADO
        # =============================
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.bilateralFilter(frame_gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_gray = clahe.apply(frame_gray)
        
        # =============================
        # 3. EXTRAIR APENAS A ROI PARA PROCESSAMENTO
        # =============================
        roi_gray = frame_gray[y1:y2, x1:x2]
        
        # =============================
        # 4. DETECTAR ÍRIS APENAS NA ROI
        # =============================
        roi_blur = cv2.medianBlur(roi_gray, 5)
        
        # Parâmetros otimizados para íris na região central
        circles = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,              # Resolução do acumulador
            minDist=100,         # Distância mínima entre círculos
            param1=100,          # Threshold para detecção de bordas
            param2=30,           # Threshold para centro do círculo
            minRadius=30,        # Raio mínimo esperado para íris
            maxRadius=120         # Raio máximo esperado para íris
        )
        
        iris_center = None
        iris_radius = None
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Pega apenas o MELHOR círculo
            best_circle = circles[0][0]
            cx_roi, cy_roi, r = best_circle
            
            # Converte coordenadas da ROI para coordenadas do frame completo
            cx = cx_roi + x1
            cy = cy_roi + y1
            
            iris_center = (cx, cy)
            iris_radius = r
            
            # Desenha íris detectada
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 3)  # Círculo da íris (vermelho)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)  # Centro (verde)
            
            # Adiciona texto informativo
            cv2.putText(frame, f"Iris: r={r}px", (cx-50, cy-r-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Análise de capsulorrexe
            contour, quality = self.caps.detect_capsulorrhexis(frame, (cx, cy), r)
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
                self.metrics["capsulorrhexis_quality"].append(quality)
        
        # =============================
        # 5. DETECTAR FEATURES APENAS DENTRO DA ÍRIS
        # =============================
        mask = np.zeros_like(frame_gray)
        
        if iris_center is not None:
            # Máscara circular expandida (1.5x o raio da íris)
            cv2.circle(mask, iris_center, int(iris_radius * 1.5), 255, -1)
        else:
            # Se não detectou íris, usa ROI central como fallback
            mask[y1:y2, x1:x2] = 255
        
        # Parâmetros para detecção de features
        feature_params = dict(
            maxCorners=200,      # Menos pontos, mais focados
            qualityLevel=0.01,   # Qualidade moderada
            minDistance=10,      # Distância maior entre pontos
            blockSize=7,
            mask=mask            # Usa máscara para restringir área
        )
        
        # Inicialização no primeiro frame
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.prev_points = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
            self.frame_counter = 0
            
            results = {
                "iris_center": iris_center,
                "iris_radius": iris_radius,
                "num_points": 0,
                "motion": 0,
                "roi_bounds": (x1, y1, x2, y2)
            }
            
            if "capsulorrhexis_quality" in locals():
                results["capsulorrhexis_quality"] = quality
                
            return frame, results
        
        # =============================
        # 6. OPTICAL FLOW (Lucas-Kanade)
        # =============================
        lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        
        motion_vector = 0
        
        if self.prev_points is not None and len(self.prev_points) > 0:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                frame_gray,
                self.prev_points,
                None,
                **lk_params
            )
            
            if new_points is not None and status is not None:
                # Filtra pontos válidos
                good_new = new_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                valid_new = []
                valid_old = []
                
                # Filtra pontos dentro dos limites e dentro da ROI
                for (xn, yn), (xo, yo) in zip(good_new, good_old):
                    xn, yn = int(xn), int(yn)
                    xo, yo = int(xo), int(yo)
                    
                    # Verifica se está dentro dos limites
                    if not (0 <= xn < w and 0 <= yn < h):
                        continue
                        
                    # Verifica se está dentro da ROI ou próximo da íris
                    if iris_center is not None:
                        # Distância do ponto ao centro da íris
                        dist_to_iris = np.sqrt((xn - iris_center[0])**2 + (yn - iris_center[1])**2)
                        if dist_to_iris > iris_radius * 1.5:
                            continue  # Ignora pontos muito distantes da íris
                    else:
                        # Fallback: verifica se está na ROI
                        if not (x1 <= xn <= x2 and y1 <= yn <= y2):
                            continue
                    
                    valid_new.append((xn, yn))
                    valid_old.append((xo, yo))
                    
                    # Desenha trajetória
                    cv2.line(frame, (xn, yn), (xo, yo), (0, 255, 255), 1)
                    cv2.circle(frame, (xn, yn), 2, (255, 0, 0), -1)
                
                # Atualiza pontos para próximo frame
                if len(valid_new) > 0:
                    self.prev_points = np.array(valid_new).reshape(-1, 1, 2).astype(np.float32)
                    
                    # Calcula movimento médio
                    motion_vector = np.mean([
                        np.sqrt((xn - xo)**2 + (yn - yo)**2) 
                        for (xn, yn), (xo, yo) in zip(valid_new, valid_old)
                    ])
                else:
                    # Re-detecta features se perdeu todos os pontos
                    self.prev_points = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        
        # =============================
        # 7. REFRESH PERIÓDICO DE PONTOS
        # =============================
        self.frame_counter += 1
        
        # Re-detecta features a cada 15 frames
        if self.frame_counter % 15 == 0:
            new_pts = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
            if new_pts is not None:
                if self.prev_points is not None and len(self.prev_points) > 0:
                    # Combina pontos antigos com novos
                    combined = np.vstack((self.prev_points, new_pts))
                    if len(combined) > 200:
                        self.prev_points = combined[:200]
                    else:
                        self.prev_points = combined
                else:
                    self.prev_points = new_pts
        
        # Atualiza frame anterior
        self.prev_gray = frame_gray
        
        # =============================
        # 8. INFORMAÇÕES NA TELA
        # =============================
        num_tracked_points = len(self.prev_points) if self.prev_points is not None else 0
        
        cv2.putText(frame, f"Pontos rastreados: {num_tracked_points}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Movimento: {motion_vector:.2f}px", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        results = {
            "iris_center": iris_center,
            "iris_radius": iris_radius,
            "num_points": num_tracked_points,
            "motion": motion_vector,
            "roi_bounds": (x1, y1, x2, y2)
        }
        
        if iris_center is not None and len(self.metrics["capsulorrhexis_quality"]) > 0:
            results["capsulorrhexis_quality"] = self.metrics["capsulorrhexis_quality"][-1]
        
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
            metrics = np.zeros((350, 500, 3), dtype=np.uint8)
            cv2.putText(metrics, "METRICAS EM TEMPO REAL", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y = 60

            if "iris_center" in results and results["iris_center"] is not None:
                cx, cy = results["iris_center"]
                cv2.putText(metrics, f"Iris Centro: ({cx},{cy})", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 30

            if "iris_radius" in results and results["iris_radius"] is not None:
                cv2.putText(metrics, f"Iris Raio: {results['iris_radius']}px", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 30

            if "num_points" in results:
                cv2.putText(metrics, f"Pontos: {results['num_points']}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 30

            if "motion" in results:
                cv2.putText(metrics, f"Movimento: {results['motion']:.2f}px", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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

    print("\n=== Sistema de Análise de Cirurgia de Catarata ===")
    print("Versão Otimizada com ROI Central e Hough Circles\n")
    import time; time.sleep(1)

    analyzer = CataractSurgeryAnalyzer()

    video_path = "cirurgia.mp4"

    print(f"Processando vídeo: {video_path}")
    print("Pressione 'q' para sair\n")
    analyzer.process_video(video_path)


if __name__ == "__main__":
    main()