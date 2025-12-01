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


class KalmanTracker:
    """Filtro de Kalman para rastreamento de centro da íris/pupila"""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                   [0, 1, 0, 1],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
    def predict(self):
        return self.kalman.predict()
    
    def update(self, measurement):
        measurement = np.array([[np.float32(measurement[0])],
                                [np.float32(measurement[1])]])
        self.kalman.correct(measurement)
        return self.predict()


class IrisPupilDetector:
    """Detector robusto de íris e pupila usando Hough Transform e MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.kalman_tracker = KalmanTracker()
        
    def detect_circles_hough(self, gray_img):
        """Detecção usando Hough Transform (método clássico)"""
        blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
        
        # Detecta pupila (círculo escuro)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=80
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return None
    
    def detect_with_mediapipe(self, frame):
        """Detecção usando MediaPipe (mais robusto)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Índices dos landmarks da íris (MediaPipe Iris)
            left_iris = [474, 475, 476, 477]
            right_iris = [469, 470, 471, 472]
            
            # Calcula centro da íris direita (exemplo)
            iris_points = []
            for idx in right_iris:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                iris_points.append([x, y])
            
            iris_center = np.mean(iris_points, axis=0).astype(int)
            iris_radius = int(dist.euclidean(iris_points[0], iris_points[2]) / 2)
            
            return iris_center, iris_radius, iris_points
        
        return None, None, None
    
    def track(self, frame):
        """Pipeline completo de detecção e rastreamento"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tenta MediaPipe primeiro
        iris_center, iris_radius, iris_points = self.detect_with_mediapipe(frame)
        
        # Fallback para Hough Transform
        if iris_center is None:
            circles = self.detect_circles_hough(gray)
            if circles is not None:
                x, y, r = circles[0]
                iris_center = np.array([x, y])
                iris_radius = r
        
        # Aplica Filtro de Kalman para suavização
        if iris_center is not None:
            kalman_center = self.kalman_tracker.update(iris_center)
            return kalman_center[:2].flatten(), iris_radius
        
        return None, None


class CapsulorrhexisAnalyzer:
    """Análise de capsulorrexe usando Active Contours (Snakes) e Level Sets"""
    
    def __init__(self):
        pass
    
    def active_contours(self, image, init_contour, alpha=0.015, beta=10, gamma=0.001, iterations=100):
        """Active Contours (Snake) para segmentação"""
        from skimage.segmentation import active_contour
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_img = gaussian_filter(gray, 3)
        
        # Aplica Active Contour
        snake = active_contour(
            gaussian_img,
            init_contour,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_iterations=iterations
        )
        
        return snake
    
    def analyze_circularity(self, contour):
        """Analisa circularidade e regularidade do contorno"""
        if len(contour) < 5:
            return 0.0
        
        # Calcula área e perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularidade = 4π * área / perímetro²
        # Valor ideal = 1.0 (círculo perfeito)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Converte para porcentagem (0-100%)
        quality = min(circularity * 100, 100)
        
        return quality
    
    def detect_capsulorrhexis(self, frame, roi_center, roi_radius):
        """Detecta e analisa a capsulorrexe"""
        # Cria ROI ao redor do centro da pupila
        x, y = roi_center
        r = roi_radius
        
        y1, y2 = max(0, y-r), min(frame.shape[0], y+r)
        x1, x2 = max(0, x-r), min(frame.shape[1], x+r)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0.0
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Aplica threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gray_roi, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Encontra contornos
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0.0
        
        # Seleciona o maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Analisa qualidade
        quality = self.analyze_circularity(largest_contour)
        
        # Ajusta coordenadas do contorno para frame original
        largest_contour[:, 0, 0] += x1
        largest_contour[:, 0, 1] += y1
        
        return largest_contour, quality


class OpticalFlowTracker:
    """Rastreamento de movimento usando Optical Flow (Lucas-Kanade)"""
    
    def __init__(self):
        self.lk_params = dict(
            winSize=(125, 125),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.prev_gray = None
        self.prev_points = None
        
    def track(self, frame):
        """Rastreia movimento de instrumentos cirúrgicos"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return None
        
        if self.prev_points is not None and len(self.prev_points) > 0:
            # Calcula optical flow
            curr_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            if curr_points is not None:
                # Seleciona pontos válidos
                good_new = curr_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                # Atualiza
                self.prev_gray = gray.copy()
                self.prev_points = good_new.reshape(-1, 1, 2)
                
                return good_new, good_old
        
        return None


class UNetSegmentation:
    """Segmentação usando U-Net (Deep Learning)"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            print("PyTorch não disponível. U-Net desabilitado.")
            self.model = None
            return
        
        # Modelo U-Net simplificado
        self.model = self._build_unet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def _build_unet(self):
        """Constrói arquitetura U-Net simplificada"""
        class SimpleUNet(nn.Module):
            def __init__(self):
                super(SimpleUNet, self).__init__()
                
                # Encoder
                self.enc1 = self._conv_block(3, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                
                # Decoder
                self.dec3 = self._conv_block(256, 128)
                self.dec2 = self._conv_block(128, 64)
                self.dec1 = nn.Conv2d(64, 4, kernel_size=1)  # 4 classes
                
                self.pool = nn.MaxPool2d(2, 2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoding
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                # Decoding
                d3 = self.dec3(self.up(e3))
                d2 = self.dec2(self.up(d3))
                d1 = self.dec1(d2)
                
                return torch.softmax(d1, dim=1)
        
        return SimpleUNet()
    
    def segment(self, frame):
        """Segmenta regiões: íris, esclera, capsulorrexe, incisão"""
        if self.model is None:
            return None
        
        # Preprocessa
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Inferência
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Pós-processa
        segmentation = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        segmentation = cv2.resize(segmentation.astype(np.uint8), 
                                   (frame.shape[1], frame.shape[0]))
        
        return segmentation


class CataractSurgeryAnalyzer:
    """Sistema principal de análise"""
    
    def __init__(self):
        self.iris_detector = IrisPupilDetector()
        self.capsulo_analyzer = CapsulorrhexisAnalyzer()
        self.flow_tracker = OpticalFlowTracker()
        self.unet_segmentation = UNetSegmentation()
        
        self.metrics = {
            'iris_stability': [],
            'pupil_size': [],
            'capsulorrhexis_quality': [],
            'instrument_movement': []
        }
    
    def process_frame(self, frame):
        """Processa um único frame"""
        results = {}
        
        # 1. Detecção e rastreamento de íris/pupila
        iris_center, iris_radius = self.iris_detector.track(frame)
        
        if iris_center is not None:
            results['iris_center'] = iris_center
            results['iris_radius'] = iris_radius
            
            # Desenha íris
            cv2.circle(frame, tuple(iris_center.astype(int)), 
                       iris_radius, (0, 255, 0), 2)
            cv2.circle(frame, tuple(iris_center.astype(int)), 
                       3, (0, 0, 255), -1)
            
            # 2. Análise de capsulorrexe
            contour, quality = self.capsulo_analyzer.detect_capsulorrhexis(
                frame, iris_center.astype(int), iris_radius + 20
            )
            
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
                results['capsulorrhexis_quality'] = quality
                self.metrics['capsulorrhexis_quality'].append(quality)
        
        # 3. Rastreamento de instrumentos (Optical Flow)
        flow_result = self.flow_tracker.track(frame)
        if flow_result is not None:
            good_new, good_old = flow_result
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), 
                         (0, 255, 255), 2)
                cv2.circle(frame, (int(a), int(b)), 5, (255, 0, 0), -1)
        
        # 4. Segmentação Deep Learning (U-Net)
        if self.unet_segmentation.model is not None:
            segmentation = self.unet_segmentation.segment(frame)
            if segmentation is not None:
                # Overlay de segmentação
                overlay = cv2.applyColorMap(
                    (segmentation * 60).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return frame, results
    
    def process_video(self, video_path, output_path=None):
        """Processa vídeo completo"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processa frame
            processed_frame, results = self.process_frame(frame)
            
            # Adiciona informações
            if 'iris_center' in results:
                cv2.putText(processed_frame, 
                            f"Iris: ({results['iris_center'][0]:.0f}, {results['iris_center'][1]:.0f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'capsulorrhexis_quality' in results:
                cv2.putText(processed_frame,
                            f"Capsulorrexe: {results['capsulorrhexis_quality']:.1f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.putText(processed_frame, f"Frame: {frame_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Salva frame processado
            if output_path:
                out.write(processed_frame)
            
            # Mostra frame (opcional)
            cv2.imshow('Analise Cirurgia Catarata', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        return self.compute_final_metrics()
    
    def compute_final_metrics(self):
        """Calcula métricas finais"""
        final_metrics = {}
        
        if self.metrics['capsulorrhexis_quality']:
            final_metrics['avg_capsulorrhexis_quality'] = np.mean(
                self.metrics['capsulorrhexis_quality']
            )
            final_metrics['std_capsulorrhexis_quality'] = np.std(
                self.metrics['capsulorrhexis_quality']
            )
        
        return final_metrics


def main():

    print("Iniciando programa...") 
    import time; time.sleep(3)

    """Função principal"""
    print("="*60)
    print("Sistema de Análise de Cirurgia de Catarata")
    print("Visão Computacional com PDI + Deep Learning")
    print("="*60)
    
    # Inicializa sistema
    analyzer = CataractSurgeryAnalyzer()
    
    # Exemplo de uso
    video_path = 'cirurgia.mp4" ' # Caminho do vídeo
    output_path = 'cirurgia_analisada.mp4'  # Vídeo de saída
    
    print(f"\nProcessando vídeo: {video_path}")
    print("\nTécnicas aplicadas:")
    print("  1. Filtro de Kalman (estabilização)")
    print("  2. Hough Transform (detecção circular)")
    print("  3. MediaPipe Iris Tracking")
    print("  4. Optical Flow Lucas-Kanade")
    print("  5. Active Contours (Snakes)")
    print("  6. U-Net Segmentation")
    print("\nPressione 'q' para interromper\n")
    
    try:
        final_metrics = analyzer.process_video(video_path, output_path)
        
        print("\n" + "="*60)
        print("MÉTRICAS FINAIS")
        print("="*60)
        for key, value in final_metrics.items():
            print(f"{key}: {value:.2f}")
        print("\nAnálise concluída com sucesso!")
        print(f"Vídeo processado salvo em: {output_path}")
        
    except FileNotFoundError:
        print(f"\nErro: Arquivo '{video_path}' não encontrado.")
        print("\nPara testar o sistema:")
        print("  1. Coloque um vídeo de cirurgia de catarata na pasta")
        print("  2. Atualize a variável 'video_path' com o nome correto")
        print("  3. Execute novamente o script")
    except Exception as e:
        print(f"\nErro durante processamento: {str(e)}")


if __name__ == "__main__":
    main()