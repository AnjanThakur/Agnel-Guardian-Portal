<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/YourUsername/Agnel-Guardian-Portal">
    <img src="screenshots/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Agnel Guardian Portal</h3>
<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Stack-Full%20Stack-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Google%20Vision-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" />
  <img src="https://img.shields.io/badge/Database-MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
</p>


  <p align="center">
    Intelligent OCR & Analytics Platform for Streamlining PTA Form Management.
    <br />
    <!-- <a href="https://github.com/YourUsername/Agnel-Guardian-Portal"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <!-- <a href="https://github.com/YourUsername/Agnel-Guardian-Portal">View Demo</a>
    ·
    <a href="https://github.com/YourUsername/Agnel-Guardian-Portal/issues">Report Bug</a>
    ·
    <a href="https://github.com/YourUsername/Agnel-Guardian-Portal/issues">Request Feature</a> -->
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## 🎯 Problem Solved

Educational institutions manually enter PTA feedback forms, which is slow and error-prone.  
This system automates the entire process — reducing hours of manual work into seconds while providing structured insights.
## About The Project

## 📷 Workflow Preview

<p align="center">
  <img src="screenshots/dashboard.png" width="420"/>
  <img src="screenshots/results.png" width="420"/>
</p>

<p align="center">
  <img src="screenshots/recs.png" width="420"/>
  <img src="screenshots/analytics.png" width="420"/>
</p>

The **Agnel Guardian Portal** is a sophisticated solution designed to modernize the processing of Parent-Teacher Association (PTA) feedback forms. By leveraging advanced OCR technologies, it eliminates manual data entry, providing instant insights into departmental performance.

<p align="right">(<a href="#top">back to top</a>)</p>


## ✨ Key Features

- 📄 Automatic digitization of scanned feedback forms
- 🔍 Hybrid OCR: Google Vision (text) + OpenCV (checkbox detection)
- 📊 Real-time analytics dashboard with department performance metrics
- 🗂 Batch upload & processing pipeline
- 🧠 Structured data extraction and validation
- ⚡ FastAPI backend for high-speed processing


### Built With

This project is built using a robust, modern tech stack for performance and scalability.

*   [React](https://reactjs.org/) (Vite)
*   [FastAPI](https://fastapi.tiangolo.com/)
*   [MongoDB](https://www.mongodb.com/)
*   [Google Cloud Vision API](https://cloud.google.com/vision)
*   [OpenCV](https://opencv.org/)
*   [Tailwind CSS](https://tailwindcss.com/)
*   [Recharts](https://recharts.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

## 🧠 System Pipeline

Upload Form  
↓  
Image Preprocessing (OpenCV)  
↓  
Text Extraction (Google Vision API)  
↓  
Checkbox Detection  
↓  
Data Parsing & Validation  
↓  
MongoDB Storage  
↓  
Analytics Dashboard (React)



<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

*   Python 3.10+
*   Node.js & npm
*   MongoDB (Local or Atlas)
*   Google Cloud Service Account Key (JSON) with Vision API enabled

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/YourUsername/Agnel-Guardian-Portal.git
    cd Agnel-Guardian-Portal
    ```

2.  **Backend Setup**
    *   Create a virtual environment:
        ```sh
        python -m venv .venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        ```
    *   Install dependencies:
        ```sh
        pip install -r requirements.txt
        ```
    *   Configure Environment:
        Create a `.env` file in the root directory and add:
        ```properties
        MONGO_URI=mongodb://localhost:27017
        GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
        ```

3.  **Frontend Setup**
    *   Navigate to the frontend directory:
        ```sh
        cd frontend
        ```
    *   Install dependencies:
        ```sh
        npm install
        ```

4.  **Run the Application**
    *   Start Backend:
        ```sh
        uvicorn app.main:app --reload
        ```
    *   Start Frontend:
        ```sh
        npm run dev
        ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1.  **Login** to the portal using administrative credentials.
2.  **Upload** scanned batches of PTA forms in the "Upload" section.
3.  **Monitor** the processing status in real-time.
4.  Navigate to **Analytics** to view department-wise feedback summaries.
5.  Use **Search** to find specific student feedback forms.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] OCR Pipeline Integration (Google Vision + OpenCV)
- [x] Departmental Analytics Dashboard
- [x] Batch Upload Processing
- [ ] User Authentication & Role Management
- [ ] PDF Export for Reports
- [ ] Historical Trend Analysis

See the [open issues](https://github.com/YourUsername/Agnel-Guardian-Portal/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

**Gaurav Choithramani**

📧 gauravlan.ch@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/gaurav-ch-847552283

Project Link: [https://github.com/Gauravch-dev/Agnel-Guardian-Portal](https://github.com/Gauravch-dev/Agnel-Guardian-Portal)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   [Google Cloud Platform](https://cloud.google.com/)
*   [FastAPI Documentation](https://fastapi.tiangolo.com/)
*   [Shadcn UI](https://ui.shadcn.com/)
*   [React Icons](https://react-icons.github.io/react-icons/)

<p align="right">(<a href="#top">back to top</a>)</p>
