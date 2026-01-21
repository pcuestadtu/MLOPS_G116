# ☁️ Cloud Deployment Guide: Docker to Google Cloud Run

This guide documents the end-to-end workflow for building our FastAPI application locally and deploying it to Google Cloud Platform (GCP).

---

## 📋 Project Configuration

- **Project ID:** `dtumlops-484509`
- **Region:** `europe-west1` (Belgium)
- **Artifact Repository:** `mlops-container-images`
- **Image Name:** `my_fastapi_app`
- **Default Cloud Port:** `8080`

---

## 🛠️ Phase 1: Local Build & Publish

### Step 1: Build the Image (“The Bake”)

Compile the code and dependencies into a Docker image.

> **Important:** Run this command from the **root** of the project  
> (`~/dtu/mlops/MLOPS_G116`), **not** inside the `dockerfiles` folder.

```bash
docker build -f dockerfiles/api.dockerfile -t my_fastapi_app .
```

### Step 2: Tag the Image (“The Label”)

Rename the local image to match the destination address in Google Artifact Registry.

Format:
```bash
[Region]-docker.pkg.dev/[Project-ID]/[Repo-Name]/[Image-Name]:[Tag]
```

```bash
docker tag my_fastapi_app europe-west1-docker.pkg.dev/dtumlops-484509/mlops-container-images/ my_fastapi_app:latest
```

### Step 3: Push to Registry (“The Shipping”)

Upload the image to Google Cloud.

```bash
docker push europe-west1-docker.pkg.dev/dtumlops-484509/mlops-container-images/my_fastapi_app:latest
```

🔐 Authentication Error?

If you get an Unauthenticated error, run this once:

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

---

## 🚀 Phase 2: Deploy to Cloud Run

You can deploy the new image using the Command Line (faster) or the Web Console (easier for settings).

### Option A: Command Line (Recommended)
This single command deploys the new revision immediately.

```bash
gcloud run deploy my-fastapi-app \
  --image europe-west1-docker.pkg.dev/dtumlops-484509/mlops-container-images/my_fastapi_app:latest \
  --region europe-west1 \
  --port 8080
```

### Option B: Google Cloud Console (GUI)
1. Navigate to **Google Cloud Console > Cloud Run**.
2. Click on your service: **`my-fastapi-app`**.
3. Click **"Edit & Deploy New Revision"** (top of the page).
4. **Container Image:** Click "Select" and choose the `latest` image you just pushed.
5. **Container Port:** Ensure this is set to **`8080`**.
6. Click **Deploy**.