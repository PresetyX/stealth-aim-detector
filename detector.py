#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stealth AI Detection System
Detecta inimigos usando YOLO/ONNX com aceleração CUDA
Envia comandos via Serial para Arduino Host Shield
"""

import cv2
import numpy as np
import mss
import serial
import onnxruntime as ort
import yaml
import time
import threading
from collections import deque
from pathlib import Path


class StealthDetector:
    """Sistema de detecção com IA usando YOLO/ONNX"""
    
    def __init__(self, config_path="config.yaml"):
        """Inicializa o detector"""
        # Carrega configurações
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Configurações de captura
        self.screen_width = self.config['screen']['width']
        self.screen_height = self.config['screen']['height']
        self.capture_size = self.config['detection']['capture_size']
        
        # Área de captura centralizada
        self.capture_region = {
            'left': (self.screen_width - self.capture_size) // 2,
            'top': (self.screen_height - self.capture_size) // 2,
            'width': self.capture_size,
            'height': self.capture_size
        }
        
        # Inicializa captura de tela
        self.sct = mss.mss()
        
        # Inicializa modelo ONNX com CUDA
        self._init_model()
        
        # Inicializa comunicação Serial
        self._init_serial()
        
        # Configurações de suavização
        self.smoothing = deque(maxlen=self.config['smoothing']['window_size'])
        
        # Controle de execução
        self.running = False
        self.paused = False
        
        # Estatísticas
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
    def _init_model(self):
        """Inicializa modelo YOLO/ONNX com CUDA"""
        model_path = self.config['model']['path']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Configurações CUDA
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.config['model']['cuda_device'],
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'  # Fallback
        ]
        
        # Cria sessão ONNX
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        
        # Verifica se CUDA está ativo
        if 'CUDAExecutionProvider' in self.session.get_providers():
            print("✓ CUDA ativado com sucesso")
        else:
            print("⚠ Rodando em CPU (mais lento)")
        
        # Info do modelo
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_size = self.input_shape[2]  # Assume formato quadrado
        
    def _init_serial(self):
        """Inicializa comunicação Serial com Arduino"""
        try:
            self.serial = serial.Serial(
                port=self.config['serial']['port'],
                baudrate=self.config['serial']['baudrate'],
                timeout=0.001  # Timeout muito curto para não bloquear
            )
            time.sleep(2)  # Aguarda Arduino resetar
            print(f"✓ Serial conectada: {self.config['serial']['port']}")
        except serial.SerialException as e:
            print(f"✗ Erro ao conectar Serial: {e}")
            self.serial = None
    
    def preprocess_frame(self, frame):
        """Prepara frame para inferência YOLO"""
        # Resize mantendo aspect ratio
        img = cv2.resize(frame, (self.model_size, self.model_size))
        
        # Normaliza para [0,1] e converte BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Formato NCHW (batch, channels, height, width)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_enemies(self, frame):
        """Detecta inimigos no frame usando YOLO"""
        # Preprocessa
        input_tensor = self.preprocess_frame(frame)
        
        # Inferência
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Processa detecções (formato YOLO padrão)
        detections = self._process_detections(outputs[0])
        
        return detections
    
    def _process_detections(self, output):
        """Processa saída bruta do YOLO"""
        detections = []
        confidence_threshold = self.config['detection']['confidence_threshold']
        
        # Formato comum YOLO: [batch, num_detections, 5+num_classes]
        # [x_center, y_center, width, height, confidence, class_scores...]
        
        for detection in output[0]:  # Pega primeiro batch
            confidence = detection[4]
            
            if confidence < confidence_threshold:
                continue
            
            # Classes alvo (ajuste conforme seu modelo)
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            
            # Filtra apenas classes de inimigos
            if class_id in self.config['detection']['target_classes']:
                x_center = detection[0] * self.capture_size
                y_center = detection[1] * self.capture_size
                width = detection[2] * self.capture_size
                height = detection[3] * self.capture_size
                
                detections.append({
                    'x': x_center,
                    'y': y_center,
                    'w': width,
                    'h': height,
                    'confidence': confidence,
                    'class': class_id
                })
        
        return detections
    
    def find_best_target(self, detections):
        """Seleciona melhor alvo baseado em proximidade ao centro"""
        if not detections:
            return None
        
        center = self.capture_size / 2
        
        # Calcula distância ao centro para cada detecção
        for det in detections:
            det['distance'] = np.sqrt(
                (det['x'] - center)**2 + (det['y'] - center)**2
            )
        
        # Retorna o mais próximo do centro
        return min(detections, key=lambda d: d['distance'])
    
    def calculate_movement(self, target):
        """Calcula movimento necessário para alcançar alvo"""
        center = self.capture_size / 2
        
        # Offset do centro
        dx = target['x'] - center
        dy = target['y'] - center
        
        # Aplica sensibilidade
        sensitivity = self.config['aim']['sensitivity']
        dx *= sensitivity
        dy *= sensitivity
        
        # Aplica suavização
        if self.config['smoothing']['enabled']:
            self.smoothing.append((dx, dy))
            dx = np.mean([s[0] for s in self.smoothing])
            dy = np.mean([s[1] for s in self.smoothing])
        
        # Limita movimento máximo
        max_move = self.config['aim']['max_movement']
        distance = np.sqrt(dx**2 + dy**2)
        if distance > max_move:
            scale = max_move / distance
            dx *= scale
            dy *= scale
        
        return int(dx), int(dy)
    
    def send_movement(self, dx, dy):
        """Envia comando de movimento para Arduino via Serial"""
        if not self.serial or not self.serial.is_open:
            return False
        
        try:
            # Formato: M,dx,dy\n (compatível com seu Arduino)
            command = f"M,{dx},{dy}\n"
            self.serial.write(command.encode())
            return True
        except serial.SerialException:
            return False
    
    def capture_frame(self):
        """Captura frame da tela"""
        screenshot = self.sct.grab(self.capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def update_fps(self, frame_time):
        """Atualiza contador de FPS"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def run(self):
        """Loop principal de detecção"""
        self.running = True
        print("\n🎯 Detector iniciado!")
        print(f"Capturando {self.capture_size}x{self.capture_size} no centro da tela")
        print(f"Modelo: {self.config['model']['path']}")
        print(f"Serial: {self.config['serial']['port']}")
        print("\nPressione Ctrl+C para parar\n")
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                start_time = time.time()
                
                # Captura frame
                frame = self.capture_frame()
                
                # Detecta inimigos
                detections = self.detect_enemies(frame)
                
                # Se encontrou alvos
                if detections:
                    target = self.find_best_target(detections)
                    dx, dy = self.calculate_movement(target)
                    
                    # Envia movimento
                    if abs(dx) > 1 or abs(dy) > 1:  # Ignora movimentos muito pequenos
                        self.send_movement(dx, dy)
                        print(f"🎯 Alvo detectado | Movimento: ({dx:+4d}, {dy:+4d}) | Conf: {target['confidence']:.2f} | FPS: {self.fps:.1f}")
                else:
                    # Sem alvos, limpa suavização
                    self.smoothing.clear()
                
                # Atualiza FPS
                frame_time = time.time() - start_time
                self.update_fps(frame_time)
                
                # Controle de taxa (opcional)
                if self.config['performance']['limit_fps']:
                    target_fps = self.config['performance']['target_fps']
                    target_time = 1.0 / target_fps
                    sleep_time = target_time - frame_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Detector parado pelo usuário")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpa recursos"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
        print("✓ Recursos liberados")


def main():
    """Função principal"""
    print("""
╔═══════════════════════════════════════╗
║   STEALTH AI DETECTION SYSTEM v1.0   ║
║   YOLO + ONNX + CUDA + Serial        ║
╚═══════════════════════════════════════╝
    """)
    
    # Inicializa detector
    detector = StealthDetector(config_path="config.yaml")
    
    # Inicia detecção
    detector.run()


if __name__ == "__main__":
    main()
