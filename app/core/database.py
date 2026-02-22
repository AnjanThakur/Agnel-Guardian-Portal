from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# We default to using the Cloud database because your local MongoDB is not running right now.
# You can change this to MONGO_URI_LOCAL later if you start your local server!
MONGO_URI = os.getenv("MONGO_URI_CLOUD")

client = MongoClient(MONGO_URI)

db = client["agnel_guardian"]

feedback_forms = db["feedback_forms"]
feedback_ratings = db["feedback_ratings"]

# Student Marks Collections
students = db["students"]
student_marks = db["student_marks"]

print("Connected to MongoDB Atlas")
