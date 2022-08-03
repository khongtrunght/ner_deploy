cd fastapi
uvicorn app:app --host localhost --port 8000 &
cd ../streamlit
streamlit run app.py &