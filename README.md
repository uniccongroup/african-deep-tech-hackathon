African Deep Tech Hackathon – Malaria Detection App 🦟

Features

- Upload cell images and detect malaria infection (Parasitized or Uninfected)

- Medical recommendation and risk level output
- API access for integration
- Docker support for easy deployment

🚀 Getting Started

1. Clone the Repository

    git clone https://github.com/uniccongroup/african-deep-tech-hackathon.git
    cd african-deep-tech-hackathon

2. Install Requirements (Python 3.8+)

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt



Place your trained YOLOv8 classification model at:

    runs/classify/train4/weights/best.pt

You can change the path in the MODEL_PATH variable in app.py if needed.

4. Run the App

    python app.py

Visit http://localhost:5000 to use the web interface.

📦 Using Docker

1. Build Docker Image

    docker build -t malaria-hackathon:latest .

2. Run the Container

    docker run -d -p 5000:5000 --name malaria-app malaria-hackathon:latest

Visit http://localhost:5000

🧠 API Endpoint

POST /api/predict

Form Data:
- file: image file (.jpg, .png, etc.)

Response Example:

{
  "prediction": "Parasitized",
  "confidence": 0.98,
  "all_probabilities": {
    "Parasitized": 0.98,
    "Uninfected": 0.02
  },
  "recommendation": "...",
  "risk_level": "danger"
}

Folder Structure

.
├── app.py
├── uploads/            
├── templates/
│   ├── index.html
│   ├── result.html
│   └── about.html
├── runs/
│   └── classify/train4/weights/best.pt
└── Dockerfile

🧪 Sample Test Image

You can use test cell images from the Cell Images for Detecting Malaria dataset:


🤝 Contributors

