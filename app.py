
import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from groq import Groq

# Set up the Groq API
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# Load the open-access DistilGPT-2 model for conversational tasks (chat completion)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load SentenceTransformer model for similarity search (to match hobbies)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient model for embedding

# Expanded list of hobbies with detailed descriptions
hobbies = [
    {"hobby": "Painting", "description": "Painting allows you to express your creativity through colors. Whether using watercolors, acrylics, or oils, you can create stunning works of art that reflect your imagination, improve your focus, and boost your mood."},
    {"hobby": "Gardening", "description": "Gardening involves growing plants, flowers, and vegetables. It helps you connect with nature, promote mental well-being, and can even provide fresh produce for your meals."},
    {"hobby": "Reading", "description": "Reading books, articles, or novels opens up new worlds, improves your vocabulary, and enhances comprehension skills. It can also foster empathy by immersing you in diverse stories and perspectives."},
    {"hobby": "Photography", "description": "Photography captures moments in time. It can be both a creative outlet and a way to preserve memories. Learn about composition, lighting, and editing to create beautiful photos."},
    {"hobby": "Cooking", "description": "Cooking allows you to explore different cuisines and develop your culinary skills. It's a rewarding hobby that combines creativity with practicality, and can be a fun activity to do alone or with loved ones."},
    {"hobby": "Playing Guitar", "description": "Learning to play the guitar allows you to create music and express your emotions. Whether you're strumming chords or mastering solos, playing the guitar enhances coordination and relieves stress."},
    {"hobby": "Coding", "description": "Coding or programming lets you create software, websites, or apps. It teaches problem-solving, logical thinking, and can lead to exciting career opportunities in tech."},
    {"hobby": "Web Development", "description": "Web development involves building and maintaining websites. It combines creativity with technical skills, allowing you to create engaging and interactive websites using languages like HTML, CSS, JavaScript, and more."},
    {"hobby": "Finance Management", "description": "Finance management teaches you how to manage your money, investments, and savings effectively. Understanding finance can help you make informed decisions about budgeting, investing, and achieving financial goals."},
    {"hobby": "Cryptocurrency", "description": "Cryptocurrency involves digital currencies like Bitcoin and Ethereum. Learn about blockchain technology, how cryptocurrencies are mined, and how to securely trade and invest in them."},
    {"hobby": "Stock Market Analysis", "description": "Stock market analysis helps you understand how the stock market works. Learn to analyze stocks, market trends, and make informed investment decisions for building wealth."},
    {"hobby": "Playing Soccer", "description": "Soccer is a fast-paced team sport that helps improve cardiovascular health, coordination, and teamwork skills. Playing soccer promotes fitness, and it’s a great way to socialize with others."},
    {"hobby": "Running", "description": "Running is an excellent cardiovascular exercise. It improves overall health, builds stamina, and is a great way to relieve stress. Plus, you can run anywhere, making it an accessible hobby."},
    {"hobby": "Cycling", "description": "Cycling provides a low-impact way to stay fit while exploring the outdoors. It's an eco-friendly mode of transportation and a fun way to enjoy nature while getting a full-body workout."},
    {"hobby": "Chess", "description": "Chess is a strategy game that improves your critical thinking, problem-solving, and patience. Whether you're a beginner or an expert, it's a game that can be played at any level and offers endless challenges."},
    {"hobby": "Video Gaming", "description": "Video gaming can be a fun and immersive way to relax and unwind. From single-player adventures to multiplayer competitions, gaming allows you to explore virtual worlds, enhance strategic thinking, and connect with others."},
    {"hobby": "Traveling", "description": "Traveling exposes you to new cultures, languages, and landscapes. It's an exciting way to broaden your horizons, meet new people, and experience the world in a way that can't be replicated through books or media."},
    {"hobby": "Yoga", "description": "Yoga combines physical poses with breathing exercises to promote flexibility, strength, and mental clarity. It’s a holistic approach to health that can improve both your body and mind."},
    {"hobby": "Fitness Training", "description": "Fitness training involves working out to enhance physical strength, endurance, and overall well-being. It can include activities like weightlifting, cardio exercises, and flexibility training."},
    {"hobby": "Writing", "description": "Writing lets you express your thoughts and ideas creatively. Whether you're writing poetry, short stories, or journaling, it can be a therapeutic and fulfilling way to communicate and document your experiences."},
    {"hobby": "Blogging", "description": "Blogging allows you to share your insights, opinions, and experiences with the world. You can create a blog on topics you're passionate about, whether it's technology, lifestyle, or travel."},
    {"hobby": "Learning Languages", "description": "Learning a new language opens doors to different cultures and enhances communication skills. Whether for travel, work, or personal growth, knowing multiple languages can be both rewarding and practical."},
    {"hobby": "Drawing", "description": "Drawing is a creative activity that allows you to express your imagination through pencil, ink, or digital mediums. It can be a relaxing and therapeutic hobby while also improving your artistic skills."},
    {"hobby": "Dancing", "description": "Dancing is a fun way to stay active while expressing emotions through movement. Whether you enjoy ballet, hip-hop, or contemporary dance, dancing can boost your mood, flexibility, and coordination."},
]

# Create embeddings for the hobbies
hobby_embeddings = embedder.encode([h["hobby"] for h in hobbies])

def get_hobby_suggestions(user_input):
    # Encode the user input
    input_embedding = embedder.encode(user_input)
    # Find hobbies with the closest embeddings
    similarities = util.cos_sim(input_embedding, hobby_embeddings)
    best_match = torch.argmax(similarities).item()
    return hobbies[best_match]["hobby"], hobbies[best_match]["description"]

def chatbot_response(user_input):
    # Use Groq API with the DistilGPT-2 model for chat responses
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",  # Or any other open model available on Groq
    )
    return chat_completion.choices[0].message.content

# Streamlit app
def main():
    st.title("Hobby & Skill Finder Chatbot")
    st.write("Tell me about your interests, and I'll suggest a hobby!")

    user_input = st.text_input("Enter something you're interested in:")

    if st.button("Find Hobby"):
        if user_input:
            suggested_hobby, description = get_hobby_suggestions(user_input)
            chat_response = chatbot_response(f"Can you tell me more about {suggested_hobby}?")

            st.subheader("Suggested Hobby:")
            st.write(f"**{suggested_hobby}**: {description}")
            st.subheader("More Information:")
            st.write(chat_response)

if __name__ == "__main__":
    main()
