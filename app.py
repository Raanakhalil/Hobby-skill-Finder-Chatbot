
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
    {"hobby": "Martial Arts", "description": "Various combat sports and self-defense techniques that often include elements of discipline and physical fitness."},
    {"hobby": "Tae Kwon Do", "description": "A Korean martial art known for its emphasis on high, fast kicks."},
    {"hobby": "Karate", "description": "A Japanese martial art focusing on striking techniques using punches, kicks, and knee strikes."},
    {"hobby": "Judo", "description": "A Japanese martial art focusing on throws, joint locks, and pins."},
    {"hobby": "Fencing", "description": "A combat sport where two participants use swords to score points by touching their opponent."},
    {"hobby": "Ballet", "description": "A highly technical form of dance that emphasizes grace and precision."},
    {"hobby": "Chess", "description": "A strategic board game played between two players with the goal of checkmating the opponent's king."},
    {"hobby": "Puzzles", "description": "Activities where players piece together different parts to complete a picture or solve a problem."},
    {"hobby": "Board Games", "description": "Games that are played on a flat surface with pieces, cards, and other components."},
    {"hobby": "Card Games", "description": "Games played with a deck of cards, such as Poker, Bridge, or Solitaire."},
    {"hobby": "Role-Playing Games", "description": "Games where players assume the roles of characters in a fictional setting, often involving storytelling and dice."},
    {"hobby": "Tabletop Games", "description": "Games played on a flat surface, typically involving miniature figures, dice, and rules."},
    {"hobby": "Makeup", "description": "The art of applying cosmetic products to enhance or alter the appearance of the face."},
    {"hobby": "Nail Art", "description": "Decorating nails with intricate designs, colors, and embellishments."},
    {"hobby": "Fashion Designing", "description": "Creating original clothing and accessories as a form of art and self-expression."},
    {"hobby": "Photography", "description": "Capturing images using a camera to create visual records or artistic expressions."},
    {"hobby": "Jewelry Making", "description": "Creating wearable art using precious metals, gemstones, beads, and other materials."},
    {"hobby": "Candle Making", "description": "Crafting candles from wax and other materials, often as a hobby for relaxation or decoration."},
    {"hobby": "Soap Making", "description": "Creating soap by combining oils, lye, and other ingredients, often with added fragrances and colors."},
    {"hobby": "Pottery Painting", "description": "Decorating pottery with paint, glazes, or other mediums to create unique, personalized pieces."},
    {"hobby": "Scrapbooking", "description": "The practice of preserving memories in books using photos, paper, and other materials."},
    {"hobby": "Embroidery", "description": "Decorative stitching on fabric, often used for artistic designs, embellishments, and personalizing items."},
    {"hobby": "Leather Crafting", "description": "The art of working with leather to create items such as bags, wallets, and accessories."},
    {"hobby": "Beading", "description": "The art of making jewelry or decorative items by stringing beads together."},
    {"hobby": "T-shirt Printing", "description": "Using techniques like screen printing to transfer designs onto t-shirts."},
    {"hobby": "Origami", "description": "The Japanese art of folding paper to create decorative objects or animals."},
    {"hobby": "Cycling", "description": "Riding a bicycle for exercise, recreation, or transportation."},
    {"hobby": "Skiing", "description": "Sliding down snow-covered slopes on skis as a recreational or competitive activity."},
    {"hobby": "Snowboarding", "description": "Sliding down snow-covered slopes on a snowboard, a single wide board."},
    {"hobby": "Surfing", "description": "Riding waves on a surfboard, typically in ocean environments."},
    {"hobby": "Boating", "description": "Operating or traveling in a boat on water, either for recreation or transportation."},
    {"hobby": "Stargazing", "description": "Observing stars, planets, and celestial events in the night sky."},
    {"hobby": "Photography", "description": "Capturing images using a camera to create visual records or artistic expressions."},
    {"hobby": "Geocaching", "description": "An outdoor recreational activity where participants use GPS coordinates to hide and seek treasures."},
    {"hobby": "Canoeing", "description": "Paddling a canoe on lakes, rivers, or other bodies of water."},
    {"hobby": "Trail Running", "description": "Running on trails in nature, often over uneven terrain and in scenic environments."},
    {"hobby": "Archery", "description": "Shooting arrows with a bow at a target or for sport."},
    {"hobby": "Swimming", "description": "Moving through water by using the limbs, often for exercise or recreation."},
    {"hobby": "Yoga", "description": "A practice of physical postures, breathing exercises, and meditation to improve health and well-being."},
    {"hobby": "Pilates", "description": "A low-impact exercise method that focuses on strengthening muscles while improving posture and flexibility."},
    {"hobby": "Running", "description": "A form of cardiovascular exercise where you move at a fast pace using your legs."},
    {"hobby": "Weightlifting", "description": "The activity of lifting weights to build strength, muscle mass, and endurance."},
    {"hobby": "Basketball", "description": "A team sport played by shooting a ball through a hoop to score points."},
    {"hobby": "Soccer", "description": "A team sport where players kick a ball into the opposing team’s goal to score points."},
    {"hobby": "Tennis", "description": "A sport played with rackets and a ball, where players hit the ball over a net to score points."},
    {"hobby": "Golf", "description": "A sport where players hit a small ball into holes on a course with as few strokes as possible."},
    {"hobby": "Baseball", "description": "A bat-and-ball game played between two teams of nine players, where the objective is to score runs."},
    {"hobby": "Football", "description": "A team sport where players aim to score by carrying or kicking the ball into the opposing team's end zone."},
    {"hobby": "Volleyball", "description": "A team sport where players hit a ball over a net, aiming to land it on the opposing team’s court."},
    {"hobby": "Rugby", "description": "A team sport similar to football, but played with an oval-shaped ball."},
    {"hobby": "Boxing", "description": "A combat sport where two people fight using their fists."},
    {"hobby": "Dancing", "description": "A form of artistic expression that involves rhythmic body movement, often set to music."},
    {"hobby": "Ballet", "description": "A highly technical form of dance that emphasizes grace and precision."},
    {"hobby": "Salsa Dancing", "description": "A lively and rhythmic style of Latin dance that is often performed in couples."},
    {"hobby": "Hip-Hop Dancing", "description": "A dance style that originated in the streets and includes breakdancing, popping, and locking."},
    {"hobby": "Jazz Dancing", "description": "A lively, energetic form of dance that often involves improvisation and syncopation."},
    {"hobby": "Acting", "description": "Performing in plays, films, or television to portray characters and tell stories."},
    {"hobby": "Theatre", "description": "The performance of drama or comedy on stage for an audience."},
    {"hobby": "Stand-Up Comedy", "description": "A form of comedy where a performer delivers a solo performance to entertain an audience."},
    {"hobby": "Playing Drums", "description": "Playing percussion instruments such as the drum set to create rhythm and beats."},
    {"hobby": "Music Composition", "description": "Creating original music, often involving writing melodies, harmonies, and arranging parts for instruments."},
    {"hobby": "DJing", "description": "The art of selecting and playing music for a live audience, often in a club or party setting."},
    {"hobby": "Music Production", "description": "The process of creating and producing recorded music, often in a studio."},
    {"hobby": "Songwriting", "description": "Writing lyrics and composing music for songs."},
    {"hobby": "Opera Singing", "description": "A vocal art form that combines classical music with dramatic performances."},
    {"hobby": "Playing Cello", "description": "Playing the cello, a string instrument known for its deep and rich tones."},
    {"hobby": "Hiking", "description": "Exploring the outdoors by walking on trails, often in natural environments like forests or mountains."},
    {"hobby": "Camping", "description": "Spending time outdoors, typically in a tent or RV, often in remote or scenic locations."},
    {"hobby": "Fishing", "description": "Catching fish either as a sport or for food, typically using rods, reels, and hooks."},
    {"hobby": "Hunting", "description": "The practice of pursuing and capturing or killing wild animals for food or sport."},
    {"hobby": "Gardening", "description": "Growing plants, flowers, and vegetables for enjoyment or food production."},
    {"hobby": "Bird Watching", "description": "Observing and identifying different species of birds in their natural habitats."},
    {"hobby": "Rock Climbing", "description": "Climbing natural rock formations or artificial rock walls as a physical challenge."},
    {"hobby": "Kayaking", "description": "Paddling a small boat called a kayak through water, often in rivers or lakes."},
    {"hobby": "Woodworking", "description": "Creating or crafting objects from wood using various tools and techniques."},
    {"hobby": "Dancing", "description": "Dancing is a fun way to stay active while expressing emotions through movement. Whether you enjoy ballet, hip-hop, or contemporary dance, dancing can boost your mood, flexibility, and coordination."},
    {"hobby": "Painting", "description": "A creative activity involving the application of pigments to surfaces, allowing expression through art."},
    {"hobby": "Drawing", "description": "Creating pictures with lines, typically with pencil or pen on paper, to depict objects, scenes, or ideas."},
    {"hobby": "Pottery", "description": "Crafting items from clay, which are then hardened by firing to create functional or decorative pieces."},
    {"hobby": "Sculpture", "description": "The art of creating three-dimensional works by carving, modeling, or assembling materials."},
    {"hobby": "Knitting", "description": "A method of creating fabric by interlocking yarn with needles, often used to make clothing and accessories."},
    {"hobby": "Crocheting", "description": "A needlework technique where yarn is interlocked with a hook to create fabrics for clothing, accessories, and more."},
    {"hobby": "Quilting", "description": "A craft of sewing layers of fabric together to make a padded material, often for blankets or decorative art."},
    {"hobby": "Sewing", "description": "Using a needle and thread to stitch fabric together to create garments, accessories, or household items."},
    {"hobby": "Embroidery", "description": "Decorative stitching on fabric, often used for artistic designs, embellishments, and personalizing items."},
    {"hobby": "Beading", "description": "The art of making jewelry or decorative items by stringing beads together."},
    {"hobby": "Calligraphy", "description": "The artistic practice of beautiful handwriting or lettering."},
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
    {"hobby": "Woodworking", "description": "Creating or crafting objects from wood using various tools and techniques."},
    {"hobby": "Scrapbooking", "description": "The practice of preserving memories in books using photos, paper, and other materials."},
    {"hobby": "Origami", "description": "The Japanese art of folding paper to create decorative objects or animals."},
    {"hobby": "Candle Making", "description": "Crafting candles from wax and other materials, often as a hobby for relaxation or decoration."},
    {"hobby": "Soap Making", "description": "Creating soap by combining oils, lye, and other ingredients, often with added fragrances and colors."},
    {"hobby": "Leather Crafting", "description": "The art of working with leather to create items such as bags, wallets, and accessories."},
    {"hobby": "Pottery Painting", "description": "Decorating pottery with paint, glazes, or other mediums to create unique, personalized pieces."},
    {"hobby": "Jewelry Making", "description": "Creating wearable art using precious metals, gemstones, beads, and other materials."},
    {"hobby": "T-shirt Printing", "description": "Using techniques like screen printing to transfer designs onto t-shirts."},
    {"hobby": "Sewing", "description": "Using a needle and thread to stitch fabric together to create garments, accessories, or household items."},
    {"hobby": "Quilting", "description": "A craft of sewing layers of fabric together to make a padded material, often for blankets or decorative art."},
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
