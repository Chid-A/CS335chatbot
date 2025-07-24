import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
import torch
from geopy import distance

location_map = {
    "paris": (48.8566, 2.3522),
    "rome": (41.9028, 12.4964),
    "berlin": (52.52, 13.4050),
    "barcelona": (41.3851, 2.1734),
    "athens": (37.9838, 23.7275),
    "lisbon": (38.7169, -9.1399),
    "budapest": (47.4979, 19.0402),
    "prague": (50.0755, 14.4378),
    "krakow": (50.0647, 19.9450),
    "madrid": (40.4168, -3.7038),
    "dublin": (53.3498, -6.2603),
    "edinburgh": (55.9533, -3.1883),
    "venice": (45.4408, 12.3155),
    "milan": (45.4642, 9.1900),
    "seville": (37.3891, -5.9845),
    "granada": (37.1773, -3.5986),
    "bavaria": (48.7904, 11.4979)

}


def distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) ** 0.5


def get_city_from_coordinates(lat, lon):
    closest_city = min(location_map, key=lambda city: distance(
        (lat, lon), location_map[city]))
    return closest_city


data = {
    'landmark': [
        'Eiffel Tower', 'Louvre Museum', 'Notre-Dame Cathedral', 'Sagrada Familia',
        'Park Güell', 'Colosseum', 'Roman Forum', 'Vatican City',
        'leaning tower of Pisa', 'Florence Cathedral', 'Buckingham Palace',
        'Tower of London', 'Westminster Abbey', 'Brandenburg Gate',
        'Reichstag Building', 'Neuschwanstein Castle', 'Acropolis of Athens',
        'Parthenon', 'Plaza de España (Seville)', 'Alhambra (Granada)',
        'St. Peter\'s Basilica', 'Sistine Chapel', 'Trevi Fountain',
        'Spanish Steps', 'Duomo di Milano', 'St. Mark\'s Basilica',
        'Doge\'s Palace', 'Grand Canal (Venice)', 'Fisherman\'s Bastion (Budapest)',
        'Buda Castle (Budapest)', 'Prague Castle', 'Charles Bridge (Prague)',
        'Wawel Castle (Krakow)', 'Old Town Square (Krakow)', 'Royal Palace of Madrid',
        'Prado Museum (Madrid)', 'Alcázar of Seville', 'Guggenheim Museum Bilbao',
        'Temple Bar (Dublin)', 'Trinity College (Dublin)', 'Edinburgh Castle',
        'Arthur\'s Seat (Edinburgh)', 'Loch Ness', 'Giant\'s Causeway',
        'Christ the Redeemer (Lisbon)', 'Belém Tower (Lisbon)', 'Jerónimos Monastery (Lisbon)',
        'Syntagma Square (Athens)', 'Temple of Olympian Zeus (Athens)', 'Delphi'
    ],
    'city': [
        'Paris', 'Paris', 'Paris', 'Barcelona', 'Barcelona', 'Rome',
        'Rome', 'Vatican City', 'Pisa', 'Florence', 'London',
        'London', 'London', 'Berlin', 'Berlin', 'Bavaria', 'Athens',
        'Athens', 'Seville', 'Granada', 'Vatican City', 'Vatican City',
        'Rome', 'Rome', 'Milan', 'Venice', 'Venice', 'Venice',
        'Budapest', 'Budapest', 'Prague', 'Prague', 'Krakow', 'Krakow',
        'Madrid', 'Madrid', 'Seville', 'Bilbao', 'Dublin', 'Dublin',
        'Edinburgh', 'Edinburgh', 'Scotland', 'Northern Ireland', 'Lisbon', 'Lisbon', 'Lisbon',
        'Athens', 'Athens', 'Greece'
    ],
    'country': [
        'France', 'France', 'France', 'Spain', 'Spain', 'Italy',
        'Italy', 'Vatican City', 'Italy', 'Italy', 'United Kingdom',
        'United Kingdom', 'United Kingdom', 'Germany', 'Germany', 'Germany', 'Greece',
        'Greece', 'Spain', 'Spain', 'Vatican City', 'Vatican City',
        'Italy', 'Italy', 'Italy', 'Italy', 'Italy', 'Italy',
        'Hungary', 'Hungary', 'Czech Republic', 'Czech Republic', 'Poland', 'Poland',
        'Spain', 'Spain', 'Spain', 'Spain', 'Ireland', 'Ireland',
        'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'Portugal', 'Portugal', 'Portugal',
        'Greece', 'Greece', 'Greece'
    ],
    'latitude': [
        48.8584, 48.8606, 48.8530, 41.4036, 41.4134, 41.8902,
        41.8954, 41.9029, 43.7230, 43.7738, 51.5014,
        51.5081, 51.4994, 52.5163, 52.5186, 47.5576, 37.9715,
        37.9716, 37.3826, 37.1773, 41.9022, 41.9029,
        41.9059, 41.9046, 45.4642, 45.4344, 45.4340,
        45.4408, 47.5092, 47.4955, 50.0902, 50.0875,
        50.0546, 50.0614, 40.4184, 40.4138, 37.3832,
        43.2689, 53.3452, 53.3438, 55.9486, 55.9445,
        57.3000, 55.2367, 38.6784, 38.6916, 38.6975,
        37.9754, 37.9719, 38.4814
    ],
    'longitude': [
        2.2945, 2.3376, 2.3499, 2.1743, 2.1527, 12.4922,
        12.4860, 12.4539, 10.3966, 11.2558, -0.1419,
        -0.0759, -0.1278, 13.3777, 13.3769, 10.7498, 23.7257,
        23.7258, -5.9962, -3.5981, 12.4573, 12.4539,
        12.4833, 12.4820, 9.1900, 12.3387, 12.3406,
        12.3359, 19.0467, 19.0343, 14.4019, 14.4166,
        19.9366, 19.9380, -3.7038, -3.6922, -5.9967,
        -2.9346, -6.2619, -6.2550, -3.2017, -3.1775,
        -4.4600, -6.5078, -9.1698, -9.2158, -9.1947,
        23.7364, 23.7303, 22.5019
    ],
    'description': [
        "Eiffel Tower is a wrought-iron lattice tower and iconic symbol of Paris, France.",
        "Louvre Museum is the world's largest art museum, housing masterpieces like the Mona Lisa in Paris.",
        "Notre-Dame Cathedral is a historic medieval Catholic cathedral famed for its French Gothic architecture in Paris.",
        "Sagrada Familia is an unfinished basilica designed by Antoni Gaudí, known for its unique architectural style in Barcelona.",
        "Park Güell is a public park system with colorful mosaic art and architectural elements designed by Gaudí in Barcelona.",
        "Colosseum is an ancient Roman amphitheater known for gladiatorial contests and large-scale public spectacles in Rome.",
        "Roman Forum is the central square of ancient Rome, surrounded by ruins of important government buildings.",
        "Vatican City is the world's smallest independent state and the spiritual center of the Roman Catholic Church.",
        "Leaning Tower of Pisa is a freestanding bell tower famous worldwide for its unintended tilt in Pisa.",
        "Florence Cathedral is a Renaissance masterpiece known for its massive dome designed by Brunelleschi in Florence.",
        "Buckingham Palace is the official London residence and administrative headquarters of the British monarch.",
        "Tower of London is a historic castle and former royal palace, famous for housing the Crown Jewels.",
        "Westminster Abbey is a large Gothic church in London, known for royal coronations and burials.",
        "Brandenburg Gate is an 18th-century neoclassical monument and symbol of Berlin and German reunification.",
        "Reichstag Building is the historic German parliament building with a modern glass dome in Berlin.",
        "Neuschwanstein Castle is a 19th-century Romanesque Revival palace on a rugged hill in Bavaria.",
        "Acropolis of Athens is an ancient citadel containing iconic Greek ruins including the Parthenon.",
        "Parthenon is a former temple dedicated to the goddess Athena, atop the Acropolis in Athens.",
        "Plaza de España (Seville) is a grand semicircular plaza with Renaissance and Moorish architecture in Seville.",
        "Alhambra (Granada) is a palace and fortress complex featuring stunning Islamic architecture in Granada.",
        "St. Peter's Basilica is a Renaissance-era church and the largest Christian church in Vatican City.",
        "Sistine Chapel is renowned for its Renaissance frescoes painted by Michelangelo, inside the Vatican.",
        "Trevi Fountain is a famous Baroque fountain in Rome known for its grand design and coin tossing tradition.",
        "Spanish Steps is a monumental stairway of 135 steps linking Piazza di Spagna and Trinità dei Monti in Rome.",
        "Duomo di Milano is the cathedral church of Milan, famous for its intricate Gothic architecture.",
        "St. Mark's Basilica is a cathedral known for its opulent design, gold mosaics, and status as Venice's symbol.",
        "Doge's Palace is a Gothic palace in Venice that was the residence of the Doge and seat of government.",
        "Grand Canal (Venice) is Venice's main waterway lined with Renaissance and Gothic palaces.",
        "Fisherman's Bastion (Budapest) is a terrace with fairy-tale turrets offering panoramic views of Budapest.",
        "Buda Castle (Budapest) is a historical castle and palace complex of the Hungarian kings.",
        "Prague Castle is a large castle complex dating from the 9th century, the official office of the Czech president.",
        "Charles Bridge (Prague) is a historic stone bridge adorned with statues, crossing the Vltava River.",
        "Wawel Castle (Krakow) is a Renaissance castle residence and a symbol of Polish national identity.",
        "Old Town Square (Krakow) is a historic market square surrounded by medieval townhouses and churches.",
        "Royal Palace of Madrid is the official residence of the Spanish Royal Family, used for ceremonies.",
        "Prado Museum (Madrid) is Spain's main national art museum, housing European art masterpieces.",
        "Alcázar of Seville is a royal palace renowned for its Mudejar architecture and lush gardens.",
        "Guggenheim Museum Bilbao is a contemporary art museum famous for its innovative architecture.",
        "Temple Bar (Dublin) is a cultural quarter known for lively pubs, street performances, and nightlife.",
        "Trinity College (Dublin) is Ireland’s oldest university, home to the famous Book of Kells.",
        "Edinburgh Castle is a historic fortress dominating the skyline of Edinburgh, Scotland.",
        "Arthur's Seat (Edinburgh) is an ancient volcano offering panoramic views of the city.",
        "Loch Ness is a large freshwater lake famous for the mythical Loch Ness Monster.",
        "Giant's Causeway is a natural rock formation of hexagonal basalt columns in Northern Ireland.",
        "Christ the Redeemer (Lisbon) is a large statue of Jesus Christ overlooking Lisbon from the Sanctuary of Christ the King.",
        "Belém Tower (Lisbon) is a fortified tower symbolizing the Age of Discoveries.",
        "Jerónimos Monastery (Lisbon) is a monastery exemplifying Manueline architecture and maritime heritage.",
        "Syntagma Square (Athens) is the central square of Athens and site of the Greek Parliament.",
        "Temple of Olympian Zeus (Athens) is a colossal ruined temple dedicated to the god Zeus.",
        "Delphi is an ancient sanctuary known for the Oracle of Delphi and its archaeological significance."
    ]
}

popular_dishes_map = {
    "paris": "Croissant, Coq au Vin, Ratatouille",
    "rome": "Carbonara, Supplì, Saltimbocca",
    "berlin": "Currywurst, Pretzel, Sauerbraten",
    "barcelona": "Paella, Tapas, Crema Catalana",
    "athens": "Moussaka, Souvlaki, Spanakopita",
    "lisbon": "Bacalhau à Brás, Pastel de Nata, Caldo Verde",
    "budapest": "Goulash, Lángos, Dobos Torte",
    "prague": "Svíčková, Trdelník, Palačinky",
    "krakow": "Pierogi, Żurek, Oscypek",
    "madrid": "Tortilla Española, Churros, Cochinillo",
    "dublin": "Irish Stew, Boxty, Soda Bread",
    "edinburgh": "Haggis, Cullen Skink, Tablet",
    "venice": "Risotto al Nero di Seppia, Sarde in Saor, Fritto Misto",
    "milan": "Risotto alla Milanese, Cotoletta, Panettone",
    "seville": "Gazpacho, Jamón Ibérico, Espinacas con Garbanzos",
    "granada": "Tortilla del Sacromonte, Piononos, Remojón",
    "bavaria": "Weisswurst, Pretzel, Schweinshaxe"
}


europe_landmarks = pd.DataFrame(data)


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# Train ML Intent Classifier
training_data = [
    ("What’s nearby?", "find_nearby"),
    ("Show me attractions close to me", "find_nearby"),
    ("Landmarks near me", "find_nearby"),
    ("I want to change my location", "change_location"),
    ("I am in rome", "change_location"),
    ("I am in berlin", "change_location"),
    ("I am in paris", "change_location"),
    ("Update my coordinates", "change_location"),
    ("Tell me about the Eiffel Tower", "landmark_info"),
    ("Give me info about Colosseum", "landmark_info"),
    ("exit", "exit"),
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("What are popular dishes in Paris?", "popular_food"),
    ("Tell me about food in Rome", "popular_food"),
    ("What should I eat in Barcelona?", "popular_food"),
    ("Famous dishes in Berlin", "popular_food"),
    ("Suggest some local food", "popular_food"),
    ("quit the app", "exit"),
    ("thanks", "exit"),
    ("bye", "exit")
]

X_train, y_train = zip(*training_data)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)

# Intent Detection using ML


def detect_intent_ml(user_input):
    user_vec = vectorizer.transform([user_input])
    return clf.predict(user_vec)[0]


# Set Current Location
current_location = {'latitude': 48.8566, 'longitude': 2.3522}  # Paris


model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all descriptions
desc_texts = europe_landmarks['description'].tolist()
desc_embeddings = model.encode(desc_texts, convert_to_numpy=True)

# Build FAISS index
dimension = desc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(desc_embeddings))

# Map index to full text data (optional but helpful)
id_to_landmark = {
    i: {
        "landmark": row["landmark"],
        "city": row["city"],
        "country": row["country"],
        "description": row["description"]
    }
    for i, row in europe_landmarks.iterrows()
}


# Semantic Search Function

def semantic_search(query, top_n=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = []

    for i, row in europe_landmarks.iterrows():
        score = util.pytorch_cos_sim(query_embedding, row['embedding']).item()
        similarities.append(
            (score, row['landmark'], row['city'], row['country'], row['description']))

    similarities.sort(reverse=True)
    results = similarities[:top_n]

    print(f"\n🔎 Top {top_n} results for: '{query}'\n")
    for score, landmark, city, country, desc in results:
        print(f"- {landmark} ({city}, {country})")
        print(f"   {desc}")
        # print(f"  🔗 Similarity Score: {score:.2f}\n")


def extract_location(user_input):
    for loc in location_map:
        if loc in user_input.lower():
            return location_map[loc]
    return None


def find_nearby_landmarks(user_lat, user_lon, top_n=3, max_distance_km=300):
    landmarks_with_distance = []

    for _, row in europe_landmarks.iterrows():
        distance = haversine(user_lat, user_lon,
                             row["latitude"], row["longitude"])
        if distance <= max_distance_km:
            landmarks_with_distance.append(
                (distance, row['landmark'], row['city'], row['country']))

    landmarks_with_distance.sort(key=lambda x: x[0])
    nearest = landmarks_with_distance[:top_n]

    print("\n Nearby Landmarks:\n")
    for dist, landmark, city, country in nearest:
        print(f"- {landmark} in {city}, {country} ({dist:.1f} km away)")


def chat_with_bot(user_input, current_lat, current_lon):

    # Detect intent
    intent = detect_intent_ml(user_input)

    if intent == "change_location":
        new_location = extract_location(user_input)
        if new_location:
            return f"Location updated!", new_location
        else:
            return "Sorry, I couldn't recognize the location you want to change to.", (current_lat, current_lon)

    if intent == "find_nearby":
        landmarks_with_distance = []
        for _, row in europe_landmarks.iterrows():
            distance = haversine(current_lat, current_lon,
                                 row["latitude"], row["longitude"])
            if distance <= 300:
                landmarks_with_distance.append(
                    (distance, row['landmark'], row['city'], row['country']))

        landmarks_with_distance.sort(key=lambda x: x[0])
        nearest = landmarks_with_distance[:3]
        if not nearest:
            return "No nearby landmarks found.", (current_lat, current_lon)

        reply = "Nearby Landmarks:\n"
        for dist, landmark, city, country in nearest:
            reply += f"- {landmark} in {city}, {country} ({dist:.1f} km away)\n"
        return reply, (current_lat, current_lon)

    elif intent == "greeting":
        return "Hello! I’m your European Travel Assistant. Ask me about landmarks or nearby attractions.", (current_lat, current_lon)

    elif intent == "landmark_info":
        query_embedding = model.encode([user_input])
        D, I = faiss_index.search(np.array(query_embedding), k=3)

        retrieved = [id_to_landmark[idx] for idx in I[0]]
        context = "\n".join(
            [f"{item['landmark']} in {item['city']}, {item['country']}: {item['description']}" for item in retrieved])

        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

        # Use Flan-T5 (local) or OpenAI (online)
        from transformers import pipeline
        rag_generator = pipeline(
            "text2text-generation", model="google/flan-t5-base")

        result = rag_generator(prompt, max_length=100)[0]['generated_text']
        return result.strip(), (current_lat, current_lon)

    elif intent == "popular_food":
        city = get_city_from_coordinates(current_lat, current_lon)
        dishes = popular_dishes_map.get(city.lower())

        if dishes:
            return f"Popular dishes in {city.title()} include: {dishes}.", (current_lat, current_lon)
        else:
            return f"Sorry, I don't have food info for {city.title()}.", (current_lat, current_lon)

    elif intent == "exit":
        return "Safe travels!", (current_lat, current_lon)

    else:
        return {
            "reply": "Sorry, I didn't understand. Try rephrasing your question.",

        }
