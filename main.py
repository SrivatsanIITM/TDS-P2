# main.py
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import re
import base64
from dotenv import load_dotenv
from io import BytesIO
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import json
import asyncio
import logging

# Set up logging
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Optional Gemini (used for non-Wikipedia questions)
GEMINI_API_KEY = "AIzaSyCcAsFl6a-f5N6ROJadsY5D650P39cCfwA"
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    else:
        logger.warning("GEMINI_API_KEY not set. Gemini functionality disabled.")
        GEMINI_MODEL = None
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    GEMINI_MODEL = None

app = FastAPI(title="Data Analyst Agent API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if "rank" in lc: colmap[c] = "Rank"
        elif "peak" in lc: colmap[c] = "Peak"
        elif "film" in lc or "title" in lc: colmap[c] = "Film"
        elif ("worldwide" in lc and "gross" in lc) or "gross" in lc: colmap[c] = "Worldwide gross"
        elif "year" in lc or "release" in lc: colmap[c] = "Year"
    return df.rename(columns=colmap)

def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    keep = [c for c in ["Rank","Peak","Film","Worldwide gross","Year"] if c in df.columns]
    df = df[keep].copy()

    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})", expand=False).astype(float)
    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    if "Peak" in df.columns:
        df["Peak"] = pd.to_numeric(df["Peak"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    if "Film" in df.columns:
        df["Film"] = df["Film"].astype(str).str.replace(r"\[[^\]]*\]", "", regex=True).str.strip()
    if "Worldwide gross" in df.columns:
        gross = df["Worldwide gross"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["Gross_Billions"] = pd.to_numeric(gross, errors="coerce") / 1_000_000_000.0

    return df

def fetch_films_table() -> pd.DataFrame:
    logger.debug(f"Fetching table from {WIKI_URL}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(WIKI_URL, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch {WIKI_URL}: Status {resp.status_code}")
            raise HTTPException(status_code=502, detail=f"Failed to fetch Wikipedia: Status {resp.status_code}")
        try:
            tables = pd.read_html(resp.text, flavor=["lxml", "bs4"])
        except ImportError as e:
            logger.error(f"Missing lxml dependency: {str(e)}")
            raise HTTPException(status_code=500, detail="Missing optional dependency 'lxml'. Please install lxml.")
        logger.debug(f"Found {len(tables)} tables")
        candidate = None
        for t in tables:
            t_clean = _clean_table(t)
            cols = set(t_clean.columns)
            if {"Rank","Film"}.issubset(cols) and ("Peak" in cols or "Worldwide gross" in cols) and len(t_clean) >= 10:
                candidate = t_clean; break
        if candidate is None:
            for t in tables:
                t_clean = _clean_table(t)
                if {"Rank","Film"}.issubset(set(t_clean.columns)):
                    candidate = t_clean; break
        if candidate is None or candidate.empty:
            logger.error("No valid table found")
            raise HTTPException(status_code=500, detail="Could not locate films table")
        return candidate.dropna(subset=["Rank","Film"]).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error fetching table from {WIKI_URL}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching table: {str(e)}")

def make_scatterplot_data_uri(df: pd.DataFrame) -> str:
    if not {"Rank","Peak"}.issubset(df.columns): 
        logger.warning("Rank or Peak column missing for scatterplot")
        return "Error: Rank or Peak column missing"
    data = df[["Rank","Peak"]].dropna()
    if len(data) < 2: 
        logger.warning("Insufficient data for scatterplot")
        return "Error: Insufficient data for scatterplot"
    x, y = data["Rank"].values, data["Peak"].values
    try:
        slope, intercept, *_ = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept

        fig = plt.figure(figsize=(4,3), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s=18, c="#7c3aed", alpha=0.85)
        ax.plot(line_x, line_y, "--", color="#ef4444", linewidth=2)
        ax.set_xlabel("Rank"); ax.set_ylabel("Peak")
        ax.set_title("Rank vs Peak (with regression)")
        ax.grid(True, alpha=0.3)
        buf = BytesIO(); plt.tight_layout(pad=0.9)
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        if len(b64) > 100000:
            fig = plt.figure(figsize=(3.2,2.4), dpi=90)
            ax = fig.add_subplot(111)
            ax.scatter(x, y, s=14, c="#7c3aed", alpha=0.85)
            ax.plot(line_x, line_y, "--", color="#ef4444", linewidth=1.8)
            ax.set_xlabel("Rank"); ax.set_ylabel("Peak"); ax.grid(True, alpha=0.3)
            buf = BytesIO(); plt.tight_layout(pad=0.7)
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight"); plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.error(f"Error generating scatterplot: {str(e)}")
        return f"Error: Failed to generate scatterplot - {str(e)}"

def parse_questions(text: str) -> List[str]:
    logger.debug("Parsing questions")
    lines = text.strip().split("\n")
    questions = []
    current_question = ""
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.\s+", line):
            if current_question:
                questions.append(current_question.strip())
            current_question = line
        else:
            current_question += " " + line
    if current_question:
        questions.append(current_question.strip())
    
    filtered_questions = []
    for q in questions:
        q_clean = re.sub(r"^\d+\.\s+", "", q).strip()
        q_lower = q_clean.lower()
        if not ("scrape" in q_lower and "wikipedia" in q_lower):
            filtered_questions.append(q_clean)
    
    logger.debug(f"Parsed questions: {filtered_questions}")
    return filtered_questions if filtered_questions else ["No valid questions found"]

@app.get("/")
def health(): return {"status": "ok", "name": "Data Analyst Agent API"}

@app.post("/api/")
async def analyze(request: Request) -> List[Dict[str, str]]:
    """
    Accepts multipart/form-data with any field names.
    - If questions mention the Wikipedia films task, performs scraping + analysis.
    - Otherwise (and if GEMINI_API_KEY is set), uses Gemini to answer.
    Returns: JSON array of objects [{"question": str, "answer": str}, ...]
    """
    logger.debug("Starting analyze endpoint")
    start_time = asyncio.get_event_loop().time()

    try:
        form = await request.form()
        logger.debug(f"Form fields: {list(form.keys())}")
    except Exception as e:
        logger.error(f"Invalid form data: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid form data")

    questions_text = None
    if form:
        for _, v in form.multi_items():
            try:
                from starlette.datastructures import UploadFile as StarUploadFile
            except Exception:
                StarUploadFile = None
            if StarUploadFile and isinstance(v, StarUploadFile):
                fname = (v.filename or "").lower()
                ctype = (v.content_type or "").lower()
                if fname.endswith(".txt") or "question" in fname or ctype == "text/plain":
                    questions_text = (await v.read()).decode("utf-8", errors="ignore")
                    logger.debug(f"Read questions.txt: {questions_text[:100]}...")
                else:
                    _ = await v.read()

    if not questions_text:
        logger.error("No questions.txt provided")
        raise HTTPException(status_code=400, detail="No questions.txt provided")

    questions = parse_questions(questions_text)
    if not questions or questions == ["No valid questions found"]:
        logger.error("No valid questions found")
        raise HTTPException(status_code=400, detail="No valid questions found")

    text_lower = questions_text.lower()
    df = None
    if "wikipedia.org/wiki/list_of_highest-grossing_films" in text_lower or "highest grossing films" in text_lower:
        try:
            df = fetch_films_table()
        except Exception as e:
            logger.error(f"Error fetching films table: {str(e)}")
            return [{"question": q, "answer": f"Error fetching data: {str(e)}"} for q in questions]

    if not GEMINI_MODEL:
        logger.warning("Gemini not available")
        return [{"question": q, "answer": "Error: Gemini not available. Please configure GEMINI_API_KEY."} for q in questions]

    # Handle scatterplot question separately
    answers = []
    other_questions = []
    for q in questions:
        if "scatterplot" in q.lower() and "rank" in q.lower() and "peak" in q.lower() and df is not None:
            scatterplot_b64 = make_scatterplot_data_uri(df)
            answers.append({"question": q, "answer": scatterplot_b64})
        else:
            other_questions.append(q)

    if not other_questions:
        return answers

    try:
        prompt_context = "You are a data analyst. Answer the following questions."
        if df is not None:
            prompt_context += " Use the following data, provided in CSV format, to answer the questions:\n\n" + df.to_csv(index=False)

        prompt = f"""
{prompt_context}

Return a JSON array of objects with 'question' and 'answer' fields: [{{"question": str, "answer": str}}, ...].
Handle errors gracefully, returning 'Error' for failed answers.

Questions:
{json.dumps(other_questions)}
"""
        async def gemini_call():
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            try:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(GEMINI_MODEL.generate_content, prompt),
                    timeout=60
                )
                raw = resp.text or ""
                logger.debug(f"Raw Gemini response: {raw[:1000]}...")
                cleaned = re.sub(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", r"\1", raw, flags=re.MULTILINE).strip()
                logger.debug(f"Cleaned Gemini response: {cleaned[:1000]}...")
                try:
                    json_match = re.search(r"\[\s*{[\s\S]*?}\s*\]", cleaned)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        if not isinstance(parsed, list):
                            logger.error("Gemini response is not a JSON array")
                            return [{"question": q, "answer": "Error: Response is not a JSON array"} for q in other_questions]
                        parsed_answers = []
                        for i, q in enumerate(other_questions):
                            answer = "Error: Missing answer"
                            if i < len(parsed) and isinstance(parsed[i], dict) and "answer" in parsed[i]:
                                answer = str(parsed[i]["answer"]).strip()
                            parsed_answers.append({"question": q, "answer": answer})
                        return parsed_answers
                    else:
                        logger.warning("No JSON array found, attempting plain text parsing")
                        lines = cleaned.split("\n")
                        parsed_answers = []
                        for i, q in enumerate(other_questions):
                            answer = "Error: No answer found"
                            for line in lines:
                                if line.startswith(f"{i+1}. "):
                                    answer = line[len(f"{i+1}. "):].strip()
                                    break
                            parsed_answers.append({"question": q, "answer": answer})
                        return parsed_answers
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    lines = cleaned.split("\n")
                    parsed_answers = []
                    for i, q in enumerate(other_questions):
                        answer = "Error: JSON parsing failed"
                        for line in lines:
                            if line.startswith(f"{i+1}. "):
                                answer = line[len(f"{i+1}. "):].strip()
                                break
                        parsed_answers.append({"question": q, "answer": answer})
                    return parsed_answers
            except Exception as e:
                logger.error(f"Gemini call error: {str(e)}")
                return [{"question": q, "answer": f"Error: Gemini call failed - {str(e)}"} for q in other_questions]

        gemini_answers = await gemini_call()
        answers.extend(gemini_answers)

    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        answers.extend([{"question": q, "answer": f"Error: {str(e)}"} for q in other_questions])

    if asyncio.get_event_loop().time() - start_time > 170:
        logger.error("Request timed out")
        raise HTTPException(status_code=500, detail="Total timeout")

    logger.debug(f"Returning answers: {[a['question'] for a in answers]}")
    return answers

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug")
