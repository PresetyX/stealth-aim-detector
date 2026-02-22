# 🎯 Stealth AI Detection System

**Sistema de detecção de inimigos com IA usando YOLO/ONNX + CUDA + Comunicação Serial**

Detector completamente novo, desenvolvido do zero em Python. Usa modelos YOLO com aceleração CUDA para detectar alvos em tempo real e envia comandos de movimento via Serial para Arduino Host Shield.

---

## 🌟 Características

- ✅ **100% Novo** - Código único, sem relação com Aimmy ou forks
- 🧠 **IA com YOLO** - Detecção precisa usando modelos ONNX
- ⚡ **Aceleração CUDA** - Inferência rápida na GPU NVIDIA
- 📡 **Comunicação Serial** - Compatível com Arduino Host Shield
- 🎮 **Invisível** - Processamento externo, sem injeção de memória
- 🔧 **Configurável** - Todos os parâmetros em arquivo YAML
- 📊 **Alta Performance** - 60+ FPS com otimizações

---

## 🛠️ Requisitos

### Hardware
- GPU NVIDIA com suporte CUDA (GTX 1050 ou superior)
- Arduino com Host Shield USB configurado
- 4GB+ RAM

### Software
- **Windows 10/11** (testado)
- **Python 3.10+**
- **CUDA Toolkit 12.x** ([Download](https://developer.nvidia.com/cuda-downloads))
- **cuDNN 9.x** ([Download](https://developer.nvidia.com/cudnn))
- **Visual C++ Redistributable** ([Download](https://aka.ms/vs/17/release/vc_redist.x64.exe))

---

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/PresetyX/stealth-aim-detector.git
cd stealth-aim-detector
```

### 2. Instale CUDA e cuDNN

**CUDA Toolkit:**
1. Baixe CUDA 12.x do [site NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Instale com configurações padrão
3. Verifique: `nvcc --version`

**cuDNN:**
1. Baixe cuDNN 9.x (requer conta NVIDIA)
2. Extraia os arquivos
3. Copie `bin`, `include`, `lib` para `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`

### 3. Instale dependências Python
```bash
pip install -r requirements.txt
```

### 4. Baixe modelo YOLO

Crie pasta `models/` e baixe um modelo ONNX:

```bash
mkdir models
```

**Opção A: Modelo pré-treinado (COCO)**
```python
from ultralytics import YOLO

# Converte YOLOv8 para ONNX
model = YOLO('yolov8n.pt')
model.export(format='onnx')
```

**Opção B: Modelo customizado**
- Treine seu próprio modelo com dados do jogo
- Use Ultralytics YOLO ou Roboflow
- Exporte para formato ONNX

### 5. Configure o Arduino

**Seu código Arduino atual NÃO precisa mudar!** Ele já recebe comandos no formato:
```
M,dx,dy\n
```

Certifique-se que:
- Arduino com Host Shield está conectado
- Porta Serial está correta (verifique no Gerenciador de Dispositivos)
- Baudrate é 115200

---

## ⚙️ Configuração

Edite `config.yaml` com suas preferências:

```yaml
# Tela
screen:
  width: 1920  # Resolução da sua tela
  height: 1080

# Detecção
detection:
  capture_size: 640  # Área de captura (maior = mais lento)
  confidence_threshold: 0.5  # Confiança mínima (0-1)
  target_classes: [0]  # Classes YOLO (0 = person no COCO)

# Serial
serial:
  port: "COM3"  # ⚠️ AJUSTE PARA SUA PORTA
  baudrate: 115200

# Mira
aim:
  sensitivity: 0.8  # Ajuste fino (0.1 = lento, 2.0 = rápido)
  max_movement: 50  # Limite de movimento por frame

# Suavização
smoothing:
  enabled: true
  window_size: 3  # Mais frames = mais suave, mas mais delay
```

---

## 🚀 Uso

### Modo Básico
```bash
python detector.py
```

### Verificar CUDA
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Deve mostrar: ['CUDAExecutionProvider', ...]
```

### Testar Serial
```python
import serial
ser = serial.Serial('COM3', 115200, timeout=1)
ser.write(b'M,10,5\n')  # Move mouse 10px direita, 5px baixo
ser.close()
```

---

## 📋 Como Funciona

```
┌─────────────┐
│   Tela      │  1. Captura região central (MSS)
│  1920x1080  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Frame 640px │  2. Preprocessa para YOLO
│  (centro)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ YOLO/ONNX   │  3. Detecta inimigos (GPU)
│  + CUDA     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Cálculo de  │  4. Calcula movimento necessário
│ Movimento   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Serial    │  5. Envia "M,dx,dy\n"
│   (USB)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Arduino    │  6. Move mouse via Host Shield
│ Host Shield │
└─────────────┘
```

---

## 🔧 Troubleshooting

### CUDA não funciona
```
⚠ Rodando em CPU (mais lento)
```

**Solução:**
1. Verifique instalação CUDA: `nvcc --version`
2. Verifique cuDNN: `find /usr -name "libcudnn*.so*"` (Linux) ou procure DLLs no Windows
3. Reinstale `onnxruntime-gpu`: `pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu`

### Serial não conecta
```
✗ Erro ao conectar Serial: [Errno 2]
```

**Solução:**
1. Verifique porta no Gerenciador de Dispositivos
2. Feche outros programas usando Serial (Arduino IDE, etc)
3. Teste porta: `python -m serial.tools.list_ports`

### FPS muito baixo (<30)

**Soluções:**
- Reduza `capture_size` em `config.yaml` (ex: 416 ou 320)
- Use modelo menor (`yolov8n` ao invés de `yolov8x`)
- Verifique se CUDA está ativo
- Feche programas pesados

### Detecções imprecisas

**Soluções:**
- Treine modelo customizado com screenshots do jogo
- Ajuste `confidence_threshold` (ex: 0.6 ou 0.7)
- Use modelo maior (`yolov8m` ou `yolov8l`)
- Ajuste `target_classes` para classes corretas

---

## 📊 Performance Esperada

| GPU                | Modelo   | FPS   | Latência |
|--------------------|----------|-------|----------|
| RTX 4090          | YOLOv8n  | 120+  | 8ms      |
| RTX 3080          | YOLOv8n  | 90+   | 11ms     |
| RTX 2060          | YOLOv8n  | 60+   | 16ms     |
| GTX 1660 Super    | YOLOv8n  | 45+   | 22ms     |
| GTX 1050 Ti       | YOLOv8n  | 30+   | 33ms     |

*Com `capture_size: 640` e CUDA ativo*

---

## 🎓 Treinar Modelo Customizado

Para melhor precisão, treine com dados do seu jogo:

### 1. Coletar dados
```python
import mss
import cv2
import time

sct = mss.mss()
for i in range(100):
    screenshot = sct.grab({'left': 640, 'top': 220, 'width': 640, 'height': 640})
    cv2.imwrite(f'dataset/images/img_{i}.png', np.array(screenshot))
    time.sleep(0.5)
```

### 2. Anotar com Roboflow
1. Crie projeto em [Roboflow](https://roboflow.com)
2. Upload das imagens
3. Anote inimigos como classe `enemy`
4. Exporte no formato YOLO

### 3. Treinar
```python
from ultralytics import YOLO

# Carrega modelo base
model = YOLO('yolov8n.pt')

# Treina
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU 0
)

# Exporta para ONNX
model.export(format='onnx')
```

### 4. Usar modelo customizado
Coloque `best.onnx` em `models/` e atualize `config.yaml`:
```yaml
model:
  path: "models/best.onnx"
```

---

## 🔐 Segurança e Ética

**⚠️ AVISO IMPORTANTE:**

Este software é fornecido **apenas para fins educacionais**. O uso de assistentes de mira em jogos online:

- ❌ Viola os Termos de Serviço da maioria dos jogos
- ❌ Pode resultar em ban permanente
- ❌ É considerado trapaça e antidesportivo

**Use apenas em:**
- ✅ Ambientes de teste offline
- ✅ Desenvolvimento e pesquisa de IA
- ✅ Jogos single-player sem anti-cheat

**O autor não se responsabiliza por:**
- Bans ou punições em jogos
- Danos a sistemas ou software
- Uso em ambientes competitivos

---

## 🧩 Diferenças do Aimmy

| Aspecto            | Aimmy (C#)      | Stealth Detector (Python) |
|--------------------|-----------------|---------------------------|
| Linguagem          | C#              | Python 100%               |
| Executável         | .exe marcado    | Script novo               |
| Histórico          | Forks conhecidos| Código original           |
| Assinatura         | Detectável      | Única                     |
| Comunicação        | Serial          | Serial (compatível)       |
| Modelo IA          | YOLO/ONNX       | YOLO/ONNX                 |
| Aceleração         | CUDA            | CUDA                      |
| Configuração       | UI              | YAML                      |

**Vantagem chave:** Código completamente novo = sem histórico detectável

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie branch para feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para branch (`git push origin feature/MinhaFeature`)
5. Abra Pull Request

---

## 📄 Licença

Este projeto é fornecido sob licença MIT. Veja `LICENSE` para mais detalhes.

**Uso educacional apenas. O autor não endossa uso em ambientes competitivos.**

---

## 📞 Suporte

- **Issues:** [GitHub Issues](https://github.com/PresetyX/stealth-aim-detector/issues)
- **Discussões:** [GitHub Discussions](https://github.com/PresetyX/stealth-aim-detector/discussions)

---

## 🙏 Agradecimentos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Framework YOLO
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Inferência otimizada
- [MSS](https://github.com/BoboTiG/python-mss) - Captura de tela rápida
- Comunidade Python e CUDA

---

<div align="center">

**Desenvolvido com 🧠 e ⚡ por Pedro**

[![GitHub](https://img.shields.io/github/stars/PresetyX/stealth-aim-detector?style=social)](https://github.com/PresetyX/stealth-aim-detector)

</div>
