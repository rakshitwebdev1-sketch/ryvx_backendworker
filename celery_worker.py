import os
import time
import random
import json
import requests
import tempfile
from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import google.generativeai as genai
import models

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

celery_app = Celery('tasks', broker=os.getenv('CELERY_BROKER_URL'), backend=os.getenv('CELERY_RESULT_BACKEND'))


@celery_app.task
def process_video_assessment(assessment_id: str):
    engine = create_engine(os.getenv("DATABASE_URL"))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    assessment = None  # Define assessment here to have it in scope for the final return
    ai_score = 0.0     # Default score

    try:
        assessment = db.query(models.SkillAssessment).filter_by(id=assessment_id).first()
        if not assessment:
            print(f"Assessment {assessment_id} not found.")
            return

        print(f"Starting REAL AI analysis for assessment {assessment_id}...")
        assessment.status = models.AssessmentStatus.processing
        db.commit()

        # --- FIX #1: DOWNLOAD THE VIDEO TO A TEMPORARY FILE ---
        video_url = assessment.video_url
        print(f"Downloading video from {video_url}...")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    temp_video_file.write(chunk)
            temp_video_path = temp_video_file.name
        
        print(f"Video downloaded to temporary path: {temp_video_path}")
        # --- END OF DOWNLOAD ---

        model = genai.GenerativeModel('gemini-1.5-flash')
        
        print(f"Uploading video from {temp_video_path} to Gemini...")
        video_file = genai.upload_file(path=temp_video_path)
        print("Video upload complete.")
        
        # Clean up the temporary file immediately after upload
        os.unlink(temp_video_path)
        
        # --- THIS IS THE FIX ---
        # We must wait for the file to be processed by Google before we can use it.
        print(f"Waiting for Gemini file to become active. File name: {video_file.name}")
        while video_file.state.name == "PROCESSING":
            print("File is still processing, waiting 5 seconds...")
            time.sleep(5)
            # Get the latest status of the file
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            raise ValueError(f"File processing failed. Final state: {video_file.state.name}")
        
        print("Gemini file is now ACTIVE and ready for use.")
        # --- END OF FIX ---
        prompt = """
        You are a senior, expert video editor and a hiring manager for an elite creative agency.
        You are reviewing a portfolio submission from a freelance video editor.
        Analyze the provided video based on the following professional criteria:
        1.  **Pacing and Rhythm:** Is the editing well-paced? Does it match the mood of the content?
        2.  **Continuity:** Are there any jarring jump cuts or continuity errors?
        3.  **Color Grading:** Is the color consistent and does it enhance the story?
        4.  **Audio Quality:** Is the audio clean, well-mixed, and free of obvious errors?
        5.  **Storytelling:** Does the edit effectively tell a clear and engaging story?

        Based on your analysis, provide a single, overall score for this editor's skill on a scale of 0.0 to 1.0, where 0.75 is the minimum passing grade for a professional.
        Then, provide a brief, one-sentence critique of the video's strongest or weakest point.

        Format your response as a JSON object with two keys: "score" (a float) and "critique" (a string).
        Example: {"score": 0.85, "critique": "Excellent pacing and rhythm, but the audio mix could be cleaner."}
        """

        print("Sending prompt to Gemini for analysis...")
        response = model.generate_content([prompt, video_file])
        print(f"--- RAW GEMINI RESPONSE ---")
        print(response.text)
        print("---------------------------")

        # 2. Add a robust check before trying to parse
        ai_response_text = response.text.strip()
        # If the response is wrapped in a markdown block, extract the JSON
        if ai_response_text.startswith("```json"):
            ai_response_text = ai_response_text[7:-3].strip() # Remove ```json and ```
    
        if not ai_response_text.startswith('{'):
            raise ValueError("AI response is not in the expected JSON format.")
        
        
        cleaned_json_text = ai_response_text.replace('`', '').replace('json', '')
        ai_data = json.loads(cleaned_json_text)
        ai_score = float(ai_data.get("score", 0.0))
        ai_critique = ai_data.get("critique", "No critique provided.")
        
        assessment.ai_score = ai_score
        assessment.human_reviewer_notes = ai_critique
        
        if ai_score >= 0.75:
            assessment.status = models.AssessmentStatus.approved
            editor = db.query(models.Editor).filter_by(id=assessment.editor_id).first()
            if editor:
                if ai_score > 0.92: editor.badge_level = 3
                elif ai_score > 0.82: editor.badge_level = 2
                else: editor.badge_level = 1
        else:
            assessment.status = models.AssessmentStatus.rejected

        db.commit()
        print(f"Finished AI analysis for assessment {assessment_id}. Score: {ai_score}, Critique: {ai_critique}")

    except Exception as e:
        print(f"An error occurred during assessment: {e}")
        if assessment:
            assessment.status = models.AssessmentStatus.rejected
            assessment.human_reviewer_notes = f"An error occurred during AI processing: {str(e)}"
            db.commit()
    finally:
        # --- FIX #2: SAFE RETURN VALUE ---
        # Get the final status safely *before* closing the session
        final_status = str(assessment.status.value) if assessment else "not_found"
        db.close()
        # --- END OF FIX ---
        
    return {"assessment_id": assessment_id, "status": final_status, "score": ai_score}