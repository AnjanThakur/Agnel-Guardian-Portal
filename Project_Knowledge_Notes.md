# 🧠 Agnel Guardian Portal — Detailed Knowledge Notes

## 1. Project Overview
- **What the project does:** It is an Intelligent OCR & Analytics Platform that automatically digitizes and processes scanned Parent-Teacher Association (PTA) feedback forms in bulk using advanced AI and computer vision.
- **Main purpose:** To eliminate the painful, slow manual entry of paper-based feedback forms by schools/universities, parsing both handwritten feedback and grid-based checkboxes instantly.
- **Real-world problem it solves:** Educational institutions collect hundreds of feedback forms from parents. Reading handwriting, entering data into excel sheets, and calculating metrics by department takes days of manual labor and is highly error-prone. This platform does it in seconds.
- **Why this project is useful:** It transforms analog, unstructured paper data into real-time, structured digital insights, generating actionable summary reports via AI for administrative decisions.

---

## 2. Core Idea and Concept
- **Main concept behind the system:** Hybrid OCR. Relying fully on a generic document AI is expensive and sometimes fails on custom checkbox grids. This system smartly separates textual data (handled by Cloud APIs/LLMs) from structural/checkbox data (handled by Computer Vision and custom ML).
- **Theory or principles used:** 
  - *Computer Vision:* Morphological transformations to detect horizontal and vertical grid lines.
  - *Deep Learning:* Use of a Convolutional Neural Network (CNN) to classify image patches as checked or unchecked.
  - *GenAI / NLP:* Using Large Language Models (LLMs) to synthesize text summaries from extracted handwritten notes.
- **Background knowledge required:** Basic understanding of Computer Vision (OpenCV contours), REST APIs, asynchronous database operations (MongoDB), and React component state.

---

## 3. Functional Explanation
- **Batch Processing & Extraction:** Teachers can upload multiple PDF forms or Image files simultaneously. The system asynchronously processes them all without freezing the frontend.
- **Automated Checkbox Detection:** Using OpenCV and a custom PyTorch model, it precisely identifies student ratings (e.g., 1 to 5) across different questions.
- **Handwriting Recognition:** Transcribes handwritten text, separating the actual "Feedback" from "Parent's contact details".
- **AI Summary Generation:** Parses all handwritten feedback and generates an automated AI summary highlighting "Key Themes", "Sentiment", and "Actionable Insights".
- **Analytics Dashboard:** A visual interface showing charts (using Recharts) on department-by-department PTA performance metrics (e.g., CSE vs IT).
- **Role-based Authentication:** Separates views based on user roles (Admin vs. Teacher vs. Parent/Student) rendering completely different portals.

---

## 4. Complete Working Logic (Step-by-Step)
1. **User Input:** A teacher logs into the React frontend and uploads a batch of scanned PDFs/images via the `FileUpload` component.
2. **Transfer:** The frontend encodes the files to Base64 strings and pushes them to the FastAPI backend via the `/ocr/pta_free` endpoint.
3. **Preprocessing:** The backend (using `pdf2image`) splits PDFs into individual frames. It uses `OpenCV` to convert the frame to grayscale and normalize lighting.
4. **Text Extraction:** The backend sends the image byte-stream to **Google Cloud Vision API**, retrieving all document text (the handwritten comments and standard form labels).
5. **Checkbox Recognition:** 
   - `grid_detector.py` scans the image for table structures, cutting out the specific rows containing the ratings.
   - The cropped rating rows are fed into `pta_rating_infer.py` (ResNet18 CNN), which looks at the check-boxes in the row and returns a probability value dictating which rating (1-5) was ticked.
6. **Data Parsing:** A custom Python Regex and split-marker heuristic (`_parse_raw_comment_text()`) looks for keywords like "Name", "Email", and "Mobile" to separate actual comments from parent contact info.
7. **Database Storage:** The parsed scores, comments, and student department details are saved securely into **MongoDB** asynchronously.
8. **AI Aggregation:** Once the batch completes, all combined comments are pushed to **Gemini API** using a strict prompt to return a structured JSON mapping the qualitative feedback.
9. **Output Generation:** The frontend receives the results, updating the local React Context (`ExtractionContext`), and visually renders the AI summary, individual scores, and data-tables for the user to review.

---

## 5. Code Structure Breakdown
The project strongly separates concerns into `frontend` and `app` (backend).

### Backend (`/app`)
- **`main.py`:** Application entry point. Bootstraps FastAPI, configures CORS, and includes routers. Serves the compiled React frontend via StaticFiles.
- **`/routes/`:** Holds API definition routes.
  - `analysis_routes.py`: Contains Gemini AI prompts (`/analysis/summarize_ai`).
  - `auth_routes.py`: Handles JWT authentication.
  - `pta_free.py`: Calls the core logic for the OCR pipeline.
- **`/ocr/`:** The brains of the image processing.
  - `pta_free_logic.py`: Glues together OpenCV preprocessing, ML inference, and Google Vision.
  - `grid_detector.py` & `table_extractor.py`: Identifies and cuts out table sections from the image matrix.
- **`/ml/`:** 
  - `pta_rating_infer.py`: Defines `PTARatingModel` utilizing `torchvision.models.resnet18` customized for finding checkmarks.
- **`/core/`:** Standard backend configurations (logging, DB connections via Motor, config loading).
- **`/models/`:** PyDantic schemas (`schemas.py`, `user_schemas.py`) to validate API JSON payloads.

### Frontend (`/frontend`)
- **`/src/components/`:** Contains React Functional Components:
  - `App.jsx`: Global router controlling restricted access paths.
  - `AnalyticsView.jsx`: Renders performance charts.
  - `FileUpload.jsx`: Dropzone for handling file selections.
  - `AISummaryReport.jsx`: Parsed display for the Gemini response.
- **`/src/context/`:** `AuthContext.jsx` and `ExtractionContext.jsx` act as the global state managers without requiring heavy frameworks like Redux.

---

## 6. Technologies Used
- **Frontend Stack:**
  - **React.js & Vite:** Core UI framework. Selected for component-based architecture; Vite was chosen for much faster Hot-Module-Reloading compared to CRA.
  - **Tailwind CSS & Shadcn UI:** Used for incredibly fast, highly beautiful utility-first styling and pre-built accessible components.
- **Backend Stack:**
  - **FastAPI:** Core backend framework. Used for its high throughput via asynchronous endpoints and built-in OpenAPI schema documentation.
  - **Motor (PyMongo):** Selected because it's a native async driver for MongoDB mapping perfectly with FastAPI's async nature.
- **AI & ML:**
  - **PyTorch (ResNet18):** Chosen because deep learning CNNs perform vastly better at visual classification tasks (like "is this box ticked?") under varying angles/lighting than classical OpenCV pixel counting.
  - **Google Cloud Vision:** The industry leader in raw handwriting (OCR) detection.
  - **Google Gemini-1.5-Flash (or local Ollama fallback):** Used because NLP LLMs are the only reliable way to summarize unstructured and sometimes grammatically broken human handwriting into cohesive executive summaries.

---

## 7. Tools, Frameworks and Libraries
- `pdf2image`: Crucial for breaking uploaded PDF batches into readable OpenCV frames.
- `opencv-python-headless`: OpenCV image manipulation (grayscale, morphological filtering).
- `pydantic`: Data validation (ensuring the frontend's API inputs match what the backend expects).
- `recharts`: A lightweight UI library used strictly to generate pie and bar charts in the frontend Analytics.
- `framer-motion`: To provide smooth CSS mount/unmount animations (like cards fading in).
- `lucide-react`: SVG icon library.

---

## 8. Architecture Explanation
- **Style:** RESTful Client-Server Architecture + Microservices-style Routing.
- **Component Relationships:** 
  - The single-page React app (Client) is entirely decoupled from the server. It speaks purely via REST API calls. 
  - Inside the backend, the architecture follows an **MVC-like (Model-View-Controller)** pattern where `models/` rule data schemas, `routes/` act as Controllers, and the Logic modules (`ocr/`, `ml/`) act as services.
- **Data Flow:**
  `Browser UI` -> JSON over HTTP -> `FastAPI HTTP Router` -> `Pydantic Validation` -> `OCR Logic Controller` -> (`Vision API Call` + `Google GenAI Call`) -> `MongoDB` -> Return HTTP 200 -> `React Global State` -> `UI Render`.

---

## 9. Database Details (MongoDB)
- **Type:** NoSQL Document Database (MongoDB). Used because forms have flexible fields that might evolve over time.
- **Structure (Collections):**
  - **`users`:** Stores admin/teachers/parents. Fields include hashed passwords, usernames, and roles.
  - **`feedbacks`:** Stores the raw analysis of forms. 
    - Purpose: So that historical processing isn't lost if the server resets. 
    - Relational data: Stores `department` and `class_name` to allow chart filtering later. Contains highly nested arrays for the ratings.

---

## 10. Algorithms and Logic Used
- **Grid Detection Algorithm (OpenCV):**
  - *Logic:* To find a table, the system applies a binary threshold, then uses a "Horizontal Kernel" (e.g. `(w, 1)`) and a "Vertical Kernel" `(1, h)` with the OpenCV `erode` and `dilate` functions. This wipes out handwriting and leaves only long solid lines behind, naturally highlighting the check-box grid skeleton.
- **ResNet-18 (CNN):**
  - *Theory:* A deep residual network that prevents the "vanishing gradient problem".
  - *Logic:* It accepts a sliced row of 5 checkboxes. It outputs an array of probabilities (`softmax`) representing which class (rating 1 to 5) has the highest likelihood of containing a physical ink mark.
- **Split-Marker NLP Heuristic:**
  - *Logic:* The system converts the whole OCR text block to lowercase and loops over an array of markers (`name:`, `mobile`, `date:`). It flags the smallest index found and cleanly divides the string into `[feedback]` and `[contact info]`.

---

## 11. API Usage
- **Google Cloud Vision (`DOCUMENT_TEXT_DETECTION`):**
  - Sends a byte-encoded PNG. Returns highly structured blocks of text strings localized by bounding boxes.
- **Gemini API / Local Ollama (`/analysis/summarize_ai` via `google.generativeai`):**
  - Takes an array of 50+ comments. Sent over with a highly strict prompt demanding a purely structured JSON response (no markdown) mapping into `executive_summary`, `key_themes`, and `actionable_insights`.

---

## 12. Data Flow Explanation
1. Raw image buffers uploaded by users enter React state.
2. The buffer is sent via `fetch` POST inside a JSON envelope `{"imageBase64": "..."}`.
3. FastAPI decodes the Base64 via `base64.b64decode` into a byte stream.
4. Converted directly into a numpy ndarray (`np.frombuffer`).
5. Processed by analytical modules.
6. A large dictionary including `{"ratings": {}, "comments": "..."}` is compiled.
7. Motor (PyMongo) asynchronously inserts this map into the NoSQL dataset.
8. The identical dictionary is beamed back to the Vite frontend and ingested into the `ResultsDisplay` component.

---

## 13. Technical Implementation Details
- **Backend Logic:** Extensively uses strict typing (`from typing import Dict, Any, List`). Protects limit usage via `bump_and_check_limit()` so users cannot spam paid external APIs. Puts high computational tasks (like `infer_rating_rows`) on the `cuda` GPU if available, else falls back to `cpu`.
- **Frontend Logic:** Built primarily on hooks (`useState`, `useEffect`, `useContext`). Prevents re-renders by wrapping large state groups into React `ContextProviders`. Protects views via a `ProtectedRoute` wrapper component checking `user.role` from the active JWT token/session.
- **Integration:** Binds Python and JS seamlessly. Due to local SPA mounting, `main.py` explicitly forces the correct MIME type for `.js` static files via `SPAStaticFiles` to prevent browser blockages on Windows.

---

## 14. Dependencies Explanation
- `motor`: Asynchronous Python driver for MongoDB.
- `fastapi`: Core routing engine.
- `google-cloud-vision` & `google-genai`: Official SDKs wrapping direct HTTP calls to Google's cloud APIs, handling auth via service accounts seamlessly.
- `opencv-python-headless`: Image matrix manipulation library (headless means it avoids downloading unnecessary window-rendering GUI dependencies like Qt).
- `torch` & `torchvision`: For defining and running the PyTorch AI model.
- `@radix-ui/*`: The engine behind Shadcn UI powering accessible dropdowns, dialogs, and tabs in the frontend.

---

## 15. System Requirements
- **Software:** Python 3.10+, Node.js (v18+), MongoDB instance (Community server or Atlas URI), Git.
- **Hardware:**
  - Standard dual-core CPU is sufficient.
  - 4GB+ RAM (Due to loading ResNet18 and processing high-res OpenCV matrices).
  - *(Optional)* NVIDIA GPU with CUDA for massive batch processing optimization in PyTorch.

---

## 16. Limitations
- **Google Quotas & Cost:** Bound by Cloud Vision pricing and usage quotas per minute.
- **Rigid Templates:** OpenCV grid detection struggles heavily if parents upload photos captured at a brutal 45-degree skew or if the table is warped and folded physically.
- **Regex Fragility:** Simple string matching for finding contact details like phone numbers (`\d{10}`) might false positive with random numeric data in comments.

---

## 17. Advantages
- Incredibly fast asynchronous pipeline allowing high throughput.
- Removes "data-entry fatigue" for school personnel.
- Highly actionable: instead of presenting 1,000 raw comments to the principal, the principal merely reads a 3-paragraph synthesized AI report representing the absolute core truths of all 1,000 parents.

---

## 18. Edge Cases
- **Blank Documents / No Checkmarks Ticked:** The CNN is trained to recognize an `empty_class=0` out of all 5 boxes and immediately flags `"status": "empty_or_noise"`.
- **Ambigous Ticks (Multiple Ticks in One Row):** Logic inspects the "Top Probability" against "Second Highest Probability". If the difference is negligible (`margin_ambiguous < 0.15`), the system marks the status as `"ambiguous"`, preventing corrupted analytic injection and telling the user to review manually.
- **API Failure:** Handled seamlessly with `try/except` fallbacks. If Gemini API is unreachable, it attempts to fall back to a local `Ollama` Llama3 model running locally using `requests`.

---

## 19. Security Aspects
- **Authentication:** `AuthContext` relies on backend verification points. Secure application routing prevents Students/Parents from manually navigating to `/admin/users` or `/extraction`.
- **API Key Masking:** All Google Cloud keys and Gemini Keys are firmly injected via environment variables (`.env`) and never exposed to the frontend payload.

---

## 20. Performance Aspects
- **Workers Queue:** To prevent overwhelming local memory and crashing the server, frontend batch processing limits HTTP outbound requests using a `CONCURRENCY_LIMIT = 5` logic block within the `runBatchOCR` function.
- **Image Resizing:** The uploaded images are immediately cropped and reshaped before being pushed through the ResNet18 tensor, drastically reducing VRAM load and saving inference time.

---

## 21. Possible Improvements
- **Technical Improvements:** Implement image deskewing (Affine Transformations/Perspective Warping) to auto-flatten folded/warped phone camera photos before extraction.
- **Features:** Direct PDF report export for the Analytics Dashboard, sending notifications to parents upon successful processing.
- **Architectural:** Migrate entirely to local Vision language models (like TrOCR or Florence-2) to cut the cost dependency on Google entirely.

---

## 22. Similar Existing Systems
- **Google Document AI / AWS Textract:** Enterprise scale equivalents. However, these are generic form parsers and they charge premium rates—they lack custom-tuned logic for Agnel institutions and lack specific workflow dashboards catering purely to academic PTA structures.
- **Typeform / Google Forms:** Purely digital methods. The Agnel portal exists precisely because physical paper/signing is still heavily mandated in certain institutional loops.

---

## 23. Keywords and Important Concepts
- **Hybrid OCR Pipeline:** Fusing classical computer vision code and modern Deep nets side-by-side.
- **OpenCV Morphological Operations:** The methodology of eroding/dilating pixels to isolate geometric structures.
- **Convolutional Neural Network (CNN - ResNet18):** Advanced machine learning algorithm specifically for interpreting 2D image data.
- **LLM Summary Synthesis:** Using AI specifically for thematic grouping of datasets.
- **Asynchronous Task Queue:** Processing heavy items without stopping the system.

---

## 24. Questions I may be asked in viva
1. **Conceptual:** *"Why did you use React Context instead of Redux?"*
   - *Answer:* For this scale, Redux's boilerplate is overkill. Context perfectly manages the global state (current user session and global file extraction tree) with zero overhead.
2. **Technical:** *"How exactly does the system know where the checkboxes are?"*
   - *Answer:* We use OpenCV's thresholding to strip away color, apply a mathematical matrix (kernel) to search uniquely for long horizontal and vertical lines, map their intersections (which perfectly bound individual cells), and then crop these cells to feed to our ML model.
3. **Design Choice:** *"Why not use Google Vision for the checkboxes too?"*
   - *Answer:* Text OCR engines look for text (characters). They fundamentally treat graphical checkboxes or hand-drawn ink ticks as irrelevant noise. To understand checkboxes accurately without false positives, we trained a specific ResNet model acting solely as a "Tick/No-Tick Classifier."
4. **Resilience:** *"What happens if the Google Vision API times out during a batch upload of 100 files?"*
   - *Answer:* The frontend manages individual Promises per file inside a bounded concurrency queue. Failed forms update local state to `"status": "error"`, leaving the successful ones untouched. You can retry the failed documents securely without reprocessing the whole batch.
