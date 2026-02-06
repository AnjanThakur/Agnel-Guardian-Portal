from pymongo import MongoClient
import os

MONGO_URI = "mongodb+srv://agnel_admin:GauravComputer@agnel-guardian-cluster.dp23zaq.mongodb.net/?appName=agnel-guardian-cluster"

client = MongoClient(MONGO_URI)

db = client["agnel_guardian"]

feedback_forms = db["feedback_forms"]
feedback_ratings = db["feedback_ratings"]

print("Connected to MongoDB Atlas")
