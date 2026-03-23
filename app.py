from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import json
import logging
import numpy as np
from collections import defaultdict
import random
import sqlite3
import requests
from functools import lru_cache
from datetime import datetime
import gzip
from io import BytesIO
import time

app = Flask(__name__)

# Performance optimizations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 86400  # 1 day cache for static files
app.config['TEMPLATES_AUTO_RELOAD'] = False  # Disable in production for performance

# Response compression
def gzip_response(response):
    """Compress response data with gzip for supported clients"""
    accept_encoding = request.headers.get('Accept-Encoding', '')

    if 'gzip' not in accept_encoding.lower():
        return response

    if (response.status_code < 200 or response.status_code >= 300 or
            'Content-Encoding' in response.headers):
        return response

    response.direct_passthrough = False

    if response.data:
        gzip_buffer = BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=gzip_buffer) as gzip_file:
            gzip_file.write(response.data)
        response.data = gzip_buffer.getvalue()
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(response.data)
        response.headers['Vary'] = 'Accept-Encoding'

    return response

@app.after_request
def add_cache_headers(response):
    """Add cache headers to responses"""
    # Don't cache error responses
    if response.status_code >= 400:
        return response

    # Apply compression
    response = gzip_response(response)

    # Add cache headers for static files
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=86400'  # 1 day
    # Add cache headers for API responses
    elif request.path.startswith('/movie/') or request.path == '/':
        response.headers['Cache-Control'] = 'public, max-age=3600'  # 1 hour

    # Add timing header for debugging
    response.headers['X-Response-Time'] = f"{time.time() - request.start_time:.4f}s"

    return response

@app.before_request
def before_request():
    """Record request start time"""
    request.start_time = time.time()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
POPULAR_KEYWORDS = [
    "Marvel", "Star Wars", "Harry Potter", "Fast & Furious", 
    "Lord of the Rings", "Avengers", "Batman", "Spider-Man", 
    "Transformers", "Pirates of the Caribbean", "Mission Impossible",
    "James Bond", "Jurassic Park", "Matrix", "Indiana Jones",
    "Superman", "X-Men", "Terminator", "Alien", "Die Hard",
    "John Wick", "Mad Max", "Toy Story", "Pixar", "Disney"
]

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", 
          "Drama", "Fantasy", "Horror", "Mystery", "Romance", 
          "Sci-Fi", "Thriller"]

DB_PATH = "movieflix.db"
CURRENT_YEAR = datetime.now().year
MIN_RATING = 6.0  # Minimum IMDb rating for popularity

# TMDB API Configuration
TMDB_API_KEY = "4195410b228df910faccc8a862f524a6"  # Replace with your actual TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Load static JSON data
MOVIE_DATA_FILE = "main_movies.json"
POSTER_DATA_FILE = "poster_data.json"

try:
    with open(MOVIE_DATA_FILE, 'r') as f:
        MOVIE_DATA = json.load(f)
    logger.info(f"Loaded {len(MOVIE_DATA)} movies from {MOVIE_DATA_FILE}")
except Exception as e:
    logger.error(f"Error loading movie data: {e}")
    MOVIE_DATA = []

try:
    with open(POSTER_DATA_FILE, 'r') as f:
        POSTER_DATA = json.load(f)
    logger.info(f"Loaded poster data from {POSTER_DATA_FILE}")
except Exception as e:
    logger.error(f"Error loading poster data: {e}")
    POSTER_DATA = []

# Global caches
q_table = defaultdict(lambda: np.zeros(len(GENRES)))
movie_vectors = {}  # Cache movie vectors
movie_cache = {}  # Cache formatted movie details


# Function definitions
@lru_cache(maxsize=1000)
def get_movie_vector_from_genres(genres_tuple):
    """Get movie vector from a tuple of genres (for caching purposes)"""
    genres = list(genres_tuple)
    return np.array([1 if g in genres else 0 for g in GENRES])

def get_movie_vector(movie):
    """Get genre vector for a movie"""
    # Check if we already have this vector cached
    imdb_id = movie.get('imdb_id') or movie.get('imdbID')
    if imdb_id and imdb_id in movie_vectors:
        return movie_vectors[imdb_id]

    if not movie:
        return np.zeros(len(GENRES))

    # Try to get genres from the movie object
    genres = []

    # Handle different ways genres might be stored
    if 'genres' in movie:
        genres_data = movie.get('genres', [])
        if isinstance(genres_data, str):
            genres = [g.strip() for g in genres_data.split(',') if g.strip()]
        elif isinstance(genres_data, list):
            genres = [genre.get('name', 'Unknown') if isinstance(genre, dict) else str(genre) for genre in genres_data]
    elif 'Genre' in movie:
        genre_str = movie.get('Genre', '')
        if genre_str and isinstance(genre_str, str):
            genres = [g.strip() for g in genre_str.split(',') if g.strip()]

    # If we couldn't get genres and have an ID, try to get movie details
    if not genres and imdb_id:
        try:
            details = get_movie_details(imdb_id)
            if details and 'Genre' in details:
                genre_str = details.get('Genre', '')
                if genre_str and isinstance(genre_str, str):
                    genres = [g.strip() for g in genre_str.split(',') if g.strip()]
        except Exception as e:
            logger.error(f"Error getting genres for {imdb_id}: {e}")

    # If still no genres, return zeros
    if not genres:
        logger.warning(f"No genre data for {imdb_id or 'unknown'}")
        vector = np.zeros(len(GENRES))
    else:
        # Use the cached function to get the vector
        vector = get_movie_vector_from_genres(tuple(genres))

    # Cache the result if we have an ID
    if imdb_id:
        movie_vectors[imdb_id] = vector

    return vector


@lru_cache(maxsize=1000)
def get_movie_details(imdb_id):
    # Check in-memory cache first
    if imdb_id in movie_cache:
        return movie_cache[imdb_id]

    # Check database cache
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT details FROM movie_cache WHERE imdb_id = ?", (imdb_id,))
            result = cursor.fetchone()
            if result:
                cached_details = json.loads(result[0])
                movie_cache[imdb_id] = cached_details
                return cached_details
    except Exception as e:
        logger.error(f"Error retrieving movie details from DB cache: {e}")

    # Check in static data
    movie = next((m for m in MOVIE_DATA if m.get('imdb_id') == imdb_id), None)

    if movie:
        # Make sure we have capitalized keys for the template
        formatted_movie = {
            'Title': movie.get('title', 'Unknown'),
            'Poster': movie.get('poster_path', movie.get('Poster', 'N/A')),
            'imdbRating': movie.get('vote_average', movie.get('imdbRating', 'N/A')),
            'Year': movie.get('release_date', '').split('-')[0] if movie.get('release_date') else movie.get('Year', 'N/A'),
            'Runtime': movie.get('runtime', 'N/A'),
            'Rated': movie.get('adult', False) and 'R' or 'PG',
            'Plot': movie.get('overview', movie.get('Plot', 'N/A')),
            'Genre': ', '.join(g.get('name', '') for g in movie.get('genres', [])) if isinstance(
                movie.get('genres', []), list) else movie.get('Genre', 'N/A'),
            'Director': movie.get('Director', 'N/A'),
            'Actors': movie.get('Actors', 'N/A'),
            'Writer': movie.get('Writer', 'N/A'),
            'Language': movie.get('Language', 'N/A'),
            'Awards': movie.get('Awards', 'N/A')
        }

        # Only process poster entries if needed
        if 'additional_backdrops' not in formatted_movie:
            poster_entries = [p for p in POSTER_DATA if any(
                b.get('file_path', '') in [movie.get('backdrop_path', ''), movie.get('poster_path', '')] for b in
                p.get('backdrops', []))]
            if poster_entries:
                formatted_movie['additional_backdrops'] = [b.get('file_path', '') for p in poster_entries for b in
                                                          p.get('backdrops', [])]

        # Cache the result
        movie_cache[imdb_id] = formatted_movie
        try:
            with sqlite3.connect(DB_PATH) as conn:
                current_time = int(datetime.now().timestamp())
                conn.execute("INSERT OR REPLACE INTO movie_cache (imdb_id, details, timestamp) VALUES (?, ?, ?)",
                            (imdb_id, json.dumps(formatted_movie), current_time))
        except Exception as e:
            logger.error(f"Error caching movie details in DB: {e}")

        return formatted_movie

    # Try fetching from OMDB API as a fallback
    try:
        params = {
            'apikey': '38cf4d5e',
            'i': imdb_id,
            'plot': 'full'
        }
        response = requests.get("http://www.omdbapi.com/", params=params, timeout=5)
        if response.status_code == 200:
            omdb_data = response.json()
            if omdb_data.get('Response') == 'True':
                movie_cache[imdb_id] = omdb_data
                # Cache in database
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        current_time = int(datetime.now().timestamp())
                        conn.execute("INSERT OR REPLACE INTO movie_cache (imdb_id, details, timestamp) VALUES (?, ?, ?)",
                                    (imdb_id, json.dumps(omdb_data), current_time))
                except Exception as e:
                    logger.error(f"Error caching OMDB data in DB: {e}")
                return omdb_data
    except Exception as e:
        logger.error(f"Failed to fetch movie {imdb_id} from OMDB: {e}")

    return {}


def get_streaming_providers(title):
    # This could be replaced with a real streaming API in the future
    services = ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max"]
    # Return random selection of 1-3 streaming services for demo purposes
    return random.sample(services, min(random.randint(1, 3), len(services)))


# Cache for formatted movies
formatted_movie_cache = {}

def format_movie(movie):
    """
    Format a raw movie object into a standardized dictionary
    """
    # Basic validation
    if not movie or not isinstance(movie, dict):
        return None

    # Try to get imdb_id from multiple possible fields
    imdb_id = movie.get('imdbID') or movie.get('imdb_id')

    # Check if we already have this movie formatted in cache
    if imdb_id and imdb_id in formatted_movie_cache:
        return formatted_movie_cache[imdb_id]

    title = movie.get('Title') or movie.get('title')

    if not imdb_id or not title:
        logger.error(f"Missing critical movie data: {movie}")
        return None

    try:
        # Create basic formatted movie object with data we already have
        formatted = {
            'title': title,
            'imdb_id': imdb_id,
            'poster': movie.get('Poster') or movie.get('poster_path', 'N/A'),
            'rating': 'N/A',
            'year': movie.get('Year') or movie.get('release_date', '').split('-')[0] if movie.get(
                'release_date') else 'N/A',
            'streaming': []
        }

        # Only get additional details if we need them
        if formatted['rating'] == 'N/A' or formatted['poster'] == 'N/A':
            # Get additional details from API or cache (this is cached internally)
            details = get_movie_details(imdb_id)

            # Update poster if needed
            if formatted['poster'] == 'N/A' and details and 'Poster' in details and details['Poster'] != 'N/A':
                formatted['poster'] = details['Poster']

            # Handle rating
            if details:
                if 'imdbRating' in details and details['imdbRating'] != 'N/A':
                    try:
                        rating = float(details['imdbRating'])
                        formatted['rating'] = str(rating)
                    except (ValueError, TypeError):
                        pass
                elif 'vote_average' in details:
                    try:
                        rating = float(details['vote_average'])
                        formatted['rating'] = str(rating)
                    except (ValueError, TypeError):
                        pass

        # Only try to get streaming if we have a valid title and it's not already set
        if title and title != 'Unknown' and not formatted['streaming']:
            try:
                formatted['streaming'] = get_streaming_providers(title)
            except Exception as e:
                logger.error(f"Error getting streaming for {title}: {e}")

        # Cache the formatted movie
        if imdb_id:
            formatted_movie_cache[imdb_id] = formatted

        return formatted

    except Exception as e:
        logger.error(f"Error in format_movie for {imdb_id}: {e}")
        return None
def save_q_table(q_table, imdb_id):
    """Save Q-values for a specific movie to the database"""
    try:
        if imdb_id not in q_table:
            logger.warning(f"Movie {imdb_id} not found in Q-table")
            return

        q_values = q_table[imdb_id].tolist()
        logger.debug(f"Saving Q-values for {imdb_id}: {q_values}")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR REPLACE INTO q_table (imdb_id, q_values) VALUES (?, ?)",
                        (imdb_id, json.dumps(q_values)))
            conn.commit()
        logger.debug(f"Successfully saved Q-table for {imdb_id}")
    except Exception as e:
        logger.error(f"Error saving Q-table for {imdb_id}: {e}")


# Cache for popular movies
popular_movies_cache = {}
POPULAR_MOVIES_CACHE_DURATION = 3600  # 1 hour in seconds

def fetch_popular_movies(limit=10, exclude_imdb_ids=None):
    """
    Fetch popular movies excluding already seen ones
    """
    if exclude_imdb_ids is None:
        exclude_imdb_ids = set()
    elif not isinstance(exclude_imdb_ids, set):
        exclude_imdb_ids = set(exclude_imdb_ids)

    # Create a cache key based on excluded IDs
    cache_key = f"popular_{limit}_{hash(frozenset(exclude_imdb_ids))}"

    # Check if we have cached results that are still valid
    current_time = int(datetime.now().timestamp())
    if cache_key in popular_movies_cache and 'timestamp' in popular_movies_cache[cache_key]:
        if current_time - popular_movies_cache[cache_key]['timestamp'] < POPULAR_MOVIES_CACHE_DURATION:
            # Filter out any newly excluded IDs
            cached_movies = popular_movies_cache[cache_key]['movies']
            filtered_movies = [m for m in cached_movies if m.get('imdb_id') not in exclude_imdb_ids]

            # If we still have enough movies after filtering, use the cache
            if len(filtered_movies) >= min(limit, len(cached_movies) - len(exclude_imdb_ids)):
                logger.debug(f"Using cached popular movies for {cache_key}")
                return filtered_movies[:limit]

    logger.debug(f"Fetching up to {limit} popular movies, excluding {len(exclude_imdb_ids)} IDs")

    # Use a subset of keywords to reduce API calls
    selected_keywords = random.sample(POPULAR_KEYWORDS, min(5, len(POPULAR_KEYWORDS)))

    # First try with selected keywords
    all_movies = []
    for keyword in selected_keywords:
        # Try getting movies for this keyword
        movies = fetch_movies(keyword)
        for movie in movies:
            if movie.get('imdbID') and movie['imdbID'] not in exclude_imdb_ids:
                # Check if this movie is already in our list (avoid duplicates)
                if not any(m.get('imdbID') == movie['imdbID'] for m in all_movies):
                    all_movies.append(movie)

    # Shuffle to get variety
    random.shuffle(all_movies)

    # Format the movies we need
    formatted_movies = []
    for movie in all_movies[:limit*2]:  # Get more than we need in case some fail formatting
        if len(formatted_movies) >= limit:
            break

        try:
            if movie.get('imdbID'):
                # Check if we already have this movie formatted in cache
                imdb_id = movie.get('imdbID')
                formatted = None

                # Check all caches for a formatted version of this movie
                for cache_data in popular_movies_cache.values():
                    if 'movies' in cache_data:
                        for cached_movie in cache_data['movies']:
                            if cached_movie.get('imdb_id') == imdb_id:
                                formatted = cached_movie
                                break
                        if formatted:
                            break

                # If not found in cache, format it
                if not formatted:
                    formatted = format_movie(movie)

                if formatted and formatted.get('poster') != 'N/A':
                    formatted_movies.append(formatted)
        except Exception as e:
            logger.error(f"Error formatting movie {movie.get('imdbID', 'unknown')}: {str(e)}")

    logger.debug(f"Returning {len(formatted_movies)} formatted movies")

    # If still no movies, try a completely different approach
    if not formatted_movies:
        logger.warning("No movies returned with standard approach, trying backup method")
        # Try a few default movies that should exist
        default_movies = ["tt0371746", "tt0944947", "tt0468569"]  # Iron Man, Game of Thrones, The Dark Knight
        for imdb_id in default_movies:
            if imdb_id not in exclude_imdb_ids:
                details = get_movie_details(imdb_id)
                if details and 'Title' in details:
                    formatted = {
                        'title': details.get('Title', 'Unknown'),
                        'imdb_id': imdb_id,
                        'poster': details.get('Poster', 'N/A'),
                        'rating': details.get('imdbRating', 'N/A'),
                        'year': details.get('Year', 'N/A'),
                        'streaming': []
                    }
                    formatted_movies.append(formatted)

    # Cache the results
    if formatted_movies:
        popular_movies_cache[cache_key] = {
            'movies': formatted_movies,
            'timestamp': current_time
        }

    return formatted_movies[:limit]


# Cache for recommendations
recommendations_cache = {}
RECOMMENDATIONS_CACHE_DURATION = 3600  # 1 hour in seconds

def recommend_movies():
    """Generate movie recommendations based on user preferences"""
    try:
        # Check if we have cached recommendations that are still valid
        current_time = int(datetime.now().timestamp())
        if 'recommendations' in recommendations_cache and 'timestamp' in recommendations_cache:
            if current_time - recommendations_cache['timestamp'] < RECOMMENDATIONS_CACHE_DURATION:
                logger.debug("Using cached recommendations")
                return recommendations_cache['recommendations']

        # Get user preferences from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT imdb_id, action FROM preferences")
            user_preferences = [(row[0], row[1]) for row in cursor.fetchall()]

        logger.debug(f"User preferences: {len(user_preferences)} items")

        # If no preferences, return popular movies
        if not user_preferences:
            logger.info("No user preferences, returning popular movies")
            popular_movies = fetch_popular_movies(10)
            # Cache the results
            recommendations_cache['recommendations'] = popular_movies
            recommendations_cache['timestamp'] = current_time
            return popular_movies

        # Get liked and disliked movies
        liked = [p[0] for p in user_preferences if p[1] == 'like']
        disliked = [p[0] for p in user_preferences if p[1] == 'dislike']

        logger.debug(f"User has liked {len(liked)} movies and disliked {len(disliked)}")

        # Load Q-table
        q_table = load_q_table()

        # Simple recommendation approach: get movies similar to liked ones
        seen_ids = set(p[0] for p in user_preferences)

        # Get candidate movies
        all_movies = fetch_popular_movies(30, exclude_imdb_ids=seen_ids)
        if not all_movies:
            logger.warning("No candidate movies found for recommendations")
            popular_movies = fetch_popular_movies(10, exclude_imdb_ids=seen_ids)
            # Cache the results
            recommendations_cache['recommendations'] = popular_movies
            recommendations_cache['timestamp'] = current_time
            return popular_movies

        # If user has liked movies, score candidates by similarity
        if liked:
            # Precompute all liked vectors at once
            liked_vectors = []
            for like_id in liked:
                try:
                    vec = get_movie_vector({'imdbID': like_id})
                    liked_vectors.append(vec)
                except Exception as e:
                    logger.error(f"Error getting vector for {like_id}: {e}")

            if liked_vectors:
                # Calculate average liked vector
                avg_liked = np.mean(liked_vectors, axis=0)

                # Precompute all movie vectors for candidates
                movie_vectors_dict = {}
                for movie in all_movies:
                    try:
                        imdb_id = movie['imdb_id']
                        movie_vectors_dict[imdb_id] = get_movie_vector({'imdbID': imdb_id})
                    except Exception as e:
                        logger.error(f"Error getting vector for {movie.get('imdb_id')}: {e}")

                # Score all candidates at once
                scored_movies = []
                for movie in all_movies:
                    try:
                        imdb_id = movie['imdb_id']
                        if imdb_id in movie_vectors_dict:
                            movie_vec = movie_vectors_dict[imdb_id]
                            similarity = np.dot(movie_vec, avg_liked)
                            scored_movies.append((similarity, movie))
                    except Exception as e:
                        logger.error(f"Error scoring movie {movie.get('imdb_id')}: {e}")

                # Sort by score and extract movies
                scored_movies.sort(key=lambda x: x[0], reverse=True)
                recommendations = [movie for _, movie in scored_movies[:10]]

                if recommendations:
                    logger.debug(f"Returning {len(recommendations)} scored recommendations")
                    # Cache the results
                    recommendations_cache['recommendations'] = recommendations
                    recommendations_cache['timestamp'] = current_time
                    return recommendations

        # If we couldn't make scored recommendations, return popular movies
        logger.debug("Returning popular movies as recommendations")
        popular_movies = all_movies[:10]
        # Cache the results
        recommendations_cache['recommendations'] = popular_movies
        recommendations_cache['timestamp'] = current_time
        return popular_movies

    except Exception as e:
        logger.error(f"Error in recommend_movies: {e}")
        # In case of error, try to return popular movies
        try:
            popular_movies = fetch_popular_movies(10)
            return popular_movies
        except:
            # If that fails too, return an empty list
            return []


def check_and_add_timestamp_column():
    """Check if timestamp column exists in movie_cache table and add it if not"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Check if the column exists
            cursor = conn.execute("PRAGMA table_info(movie_cache)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'timestamp' not in columns:
                logger.info("Adding timestamp column to movie_cache table")
                conn.execute("ALTER TABLE movie_cache ADD COLUMN timestamp INTEGER DEFAULT 0")
                conn.commit()

                # Update existing records with current timestamp
                current_time = int(datetime.now().timestamp())
                conn.execute("UPDATE movie_cache SET timestamp = ?", (current_time,))
                conn.commit()
    except Exception as e:
        logger.error(f"Error checking or adding timestamp column: {e}")

def init_db():
    # Create tables if they don't exist
    with sqlite3.connect(DB_PATH) as conn:
        # Create tables with optimized schema
        conn.execute('''CREATE TABLE IF NOT EXISTS preferences
                        (id INTEGER PRIMARY KEY,
                         imdb_id TEXT,
                         action TEXT)''')

        conn.execute('''CREATE TABLE IF NOT EXISTS q_table
                        (imdb_id TEXT PRIMARY KEY,
                         q_values TEXT)''')

        conn.execute('''CREATE TABLE IF NOT EXISTS movie_cache
                        (imdb_id TEXT PRIMARY KEY,
                         details TEXT,
                         timestamp INTEGER)''')

        # Create search cache table if it doesn't exist
        conn.execute('''CREATE TABLE IF NOT EXISTS search_cache
                        (query TEXT PRIMARY KEY,
                         results TEXT,
                         timestamp INTEGER)''')

        # Check if timestamp column exists in movie_cache table and add it if not
        conn.commit()

    check_and_add_timestamp_column()

    with sqlite3.connect(DB_PATH) as conn:
        # Add indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_imdb_id ON preferences(imdb_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_movie_cache_timestamp ON movie_cache(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_search_cache_timestamp ON search_cache(timestamp)')

        conn.commit()

    # Load q_table from database (limit to 100 most recent entries for faster startup)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT imdb_id, q_values FROM q_table ORDER BY rowid DESC LIMIT 100")
        for imdb_id, q_values in cursor:
            try:
                q_table[imdb_id] = np.array(json.loads(q_values))
            except Exception as e:
                logger.error(f"Error loading q_table for {imdb_id}: {e}")

    # Precompute movie vectors for a subset of popular movies
    # This reduces startup time while still having vectors for commonly used movies
    popular_movies = MOVIE_DATA[:500] if len(MOVIE_DATA) > 500 else MOVIE_DATA
    for movie in popular_movies:
        if movie.get('imdb_id'):
            movie_vectors[movie['imdb_id']] = get_movie_vector(movie)

    # Clean up old cache entries
    try:
        current_time = int(datetime.now().timestamp())
        with sqlite3.connect(DB_PATH) as conn:
            # Delete search cache entries older than 7 days
            conn.execute("DELETE FROM search_cache WHERE timestamp < ?", (current_time - 604800,))  # 7 days
            # Delete movie cache entries older than 30 days
            conn.execute("DELETE FROM movie_cache WHERE timestamp < ?", (current_time - 2592000,))  # 30 days
            conn.commit()
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")

    logger.info(f"Database initialized. {len(movie_vectors)} movie vectors precomputed.")
    logger.info(f"Loaded {len(q_table)} Q-table entries.")


# Routes
@app.route('/')
def home():
    # Use a timer to measure performance
    start_time = datetime.now()

    # Get personalized recommendations (this is now cached)
    recommendations = recommend_movies()

    # Make sure we have recommendations
    if not recommendations:
        recommendations = fetch_popular_movies(10)

    # Limit to 10 recommendations to reduce template rendering time
    recommendations = recommendations[:10]

    # Calculate elapsed time
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.debug(f"Generated recommendations in {elapsed:.2f} seconds")

    # Return with the correct variable name
    return render_template('recommendations.html', recommendations=recommendations)


@app.route('/swipe', methods=['GET', 'POST'])
def swipe():
    """Handle movie swiping (like/dislike)"""
    if request.method == 'POST':
        # Check if form data exists
        if 'imdb_id' not in request.form or 'action' not in request.form:
            logger.error("Missing required form data in swipe POST")
            return "Error: Missing form data", 400

        imdb_id = request.form['imdb_id']
        action = request.form['action']

        # Log the incoming data
        logger.debug(f"Received swipe: {imdb_id} - {action}")

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO preferences (imdb_id, action) VALUES (?, ?)", 
                             (imdb_id, action))
                conn.commit()
                logger.debug(f"Successfully saved swipe: {action} on {imdb_id}")

                # Update q_table for this movie
                movie_obj = {'imdbID': imdb_id}
                movie_vec = get_movie_vector(movie_obj)

                # Load q_table
                q_table = load_q_table()

                # Set reward based on action
                reward = 1 if action == 'like' else -1

                # Update Q-table values based on movie's genre vector and reward
                q_table[imdb_id] = movie_vec * reward

                # Save updated Q-table values to database
                save_q_table(q_table, imdb_id)

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return "Database error", 500

        return redirect(url_for('swipe'))

    # Get a new movie that hasn't been seen
    with sqlite3.connect(DB_PATH) as conn:
        seen_imdb_ids = [row[0] for row in conn.execute("SELECT imdb_id FROM preferences")]

    logger.debug(f"Finding a movie the user hasn't seen yet (excluded {len(seen_imdb_ids)} movies)")

    # Try to get an unseen movie - with retry mechanism
    for _ in range(3):  # Try 3 times
        movies = fetch_popular_movies(1, exclude_imdb_ids=set(seen_imdb_ids))
        if movies:
            logger.debug(f"Presenting movie for swiping: {movies[0]['title']}")
            return render_template('swipe.html', movie=movies[0])

    # If we couldn't find any new movies after 3 tries
    logger.debug("No more unseen movies available to swipe!")
    return render_template('error.html', 
                          message="You've seen all available movies! Check out the recommendations.")


@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle movie feedback (like/dislike)"""
    imdb_id = request.form.get('imdb_id')
    action = request.form.get('action')

    # Validate input
    if not imdb_id or not action:
        logger.error("Missing required parameters in feedback")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {'status': 'error', 'message': 'Missing parameters'}, 400
        return redirect(url_for('home'))

    try:
        logger.debug(f"Recording feedback: {action} for {imdb_id}")

        # Convert action value to like/dislike
        preference = 'like' if action == 'up' else 'dislike'

        # Save preference to database
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO preferences (imdb_id, action) VALUES (?, ?)",
                         (imdb_id, preference))
            conn.commit()

        # Return JSON for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {'status': 'success'}, 200

        # Redirect for regular form submissions
        return redirect(url_for('home'))

    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {'status': 'error', 'message': str(e)}, 500
        return redirect(url_for('home'))


@app.route('/search')
def search():
    # Get the search query parameter - use 'q' to match the form input name
    keyword = request.args.get('q', '').strip()
    if keyword:
        logger.debug(f"Searching for movies with keyword: {keyword}")
        movies = fetch_movies(keyword)

        # Format search results
        search_results = []
        for movie in movies[:20]:
            formatted = format_movie(movie)
            if formatted:
                search_results.append(formatted)

        logger.debug(f"Found {len(search_results)} movies matching '{keyword}'")

        # Pass both search results and keyword to template
        return render_template('search_results.html', movies=search_results, keyword=keyword)

    # Redirect to home if no search term
    return redirect(url_for('home'))


@app.route('/movie/<imdb_id>')
def movie_details(imdb_id):
    details = get_movie_details(imdb_id)
    logger.debug(f"Movie details for {imdb_id}: {details.keys() if details else 'None'}")

    if not details or ('Title' not in details and 'title' not in details):
        logger.error(f"Movie details not found for {imdb_id}")
        return render_template('error.html', message="Movie not found or details unavailable")

    # Make sure we have a properly formatted movie object with capitalized keys
    if 'Title' not in details and 'title' in details:
        details['Title'] = details['title']

    # Get streaming info
    streaming = []
    if 'Title' in details:
        streaming = get_streaming_providers(details['Title'])
        logger.debug(f"Streaming providers for {details['Title']}: {streaming}")

    # Render the details template
    return render_template('movie_details.html', movie=details, streaming=streaming)

@app.route('/debug')
def debug():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            preferences = conn.execute("SELECT * FROM preferences").fetchall()
            qt = conn.execute("SELECT imdb_id, q_values FROM q_table").fetchall()

        # Get user movies with titles
        user_movies = []
        for pref in preferences:
            movie = next((m for m in MOVIE_DATA if m.get('imdb_id') == pref[1]), None)
            if movie:
                user_movies.append({
                    'id': pref[0],
                    'imdb_id': pref[1],
                    'title': movie.get('title', 'Unknown'),
                    'action': pref[2]
                })

        # Get Q-table entries with movie titles
        q_table_entries = []
        for qt_entry in qt:
            movie = next((m for m in MOVIE_DATA if m.get('imdb_id') == qt_entry[0]), None)
            if movie:
                q_table_entries.append({
                    'imdb_id': qt_entry[0],
                    'title': movie.get('title', 'Unknown'),
                    'q_values': json.loads(qt_entry[1])
                })

        return jsonify({
            'preferences_count': len(preferences),
            'q_table_count': len(qt),
            'user_movies': user_movies,
            'q_table_entries': q_table_entries
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/reset_preferences', methods=['GET'])
def reset_preferences():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM preferences")
            conn.execute("DELETE FROM q_table")
            conn.commit()
        return redirect(url_for('home'))
    except Exception as e:
        return render_template('error.html', message=f"Error resetting preferences: {e}")

def create_q_table():
    """Initialize Q-table with zeros for all genres"""
    return defaultdict(lambda: np.zeros(len(GENRES)))

def load_q_table():
    """Load Q-table from database or create new one"""
    q_table = create_q_table()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT imdb_id, q_values FROM q_table")
            for imdb_id, q_values in cursor:
                try:
                    q_table[imdb_id] = np.array(json.loads(q_values))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error loading Q-values for {imdb_id}: {e}")
                    q_table[imdb_id] = np.zeros(len(GENRES))
    except sqlite3.Error as e:
        logger.error(f"Database error while loading Q-table: {e}")

    return q_table

def update_q_table(imdb_id, action):
    """Update Q-values for a movie based on user action"""
    q_table = load_q_table()
    movie_vector = get_movie_vector({'imdbID': imdb_id})

    # Set reward based on action
    reward = 1.0 if action == 'up' or action == 'like' else -1.0

    # Q-learning update parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor

    # Update Q-values
    old_value = q_table[imdb_id].copy()
    q_table[imdb_id] = (1 - alpha) * old_value + alpha * (reward * movie_vector)

    # Save updated Q-table to database
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR REPLACE INTO q_table (imdb_id, q_values) VALUES (?, ?)",
                        (imdb_id, json.dumps(q_table[imdb_id].tolist())))
    except sqlite3.Error as e:
        logger.error(f"Error saving Q-table for {imdb_id}: {e}")

    return q_table

def get_movie_score(movie, q_table, user_preferences):
    """Calculate a movie's score based on Q-values and user preferences"""
    imdb_id = movie.get('imdb_id')
    if not imdb_id:
        return 0

    movie_vector = get_movie_vector({'imdbID': imdb_id})
    q_score = np.sum(q_table[imdb_id]) if imdb_id in q_table else 0

    # Get average genre vector of liked movies
    liked_movies = [p[0] for p in user_preferences if p[1] == 'like']
    if liked_movies:
        liked_vectors = [get_movie_vector({'imdbID': m}) for m in liked_movies]
        avg_liked = np.mean(liked_vectors, axis=0)
        similarity_score = np.dot(movie_vector, avg_liked)
    else:
        similarity_score = 0

    return 0.7 * similarity_score + 0.3 * q_score

# Cache for search results
search_cache = {}

def fetch_movies(keyword, year=None):
    """Fetch movies from OMDB API based on keyword and optional year"""
    # Create a cache key
    cache_key = f"{keyword}_{year}" if year else keyword

    # Check if we have this search cached and it's not too old (1 hour)
    if cache_key in search_cache:
        logger.debug(f"Using cached results for '{keyword}'")
        return search_cache[cache_key]

    # Check if we have this search in the database
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "CREATE TABLE IF NOT EXISTS search_cache (query TEXT PRIMARY KEY, results TEXT, timestamp INTEGER)"
            )

            # Get cached results if they exist and are less than 24 hours old
            current_time = int(datetime.now().timestamp())
            cursor = conn.execute(
                "SELECT results FROM search_cache WHERE query = ? AND timestamp > ?", 
                (cache_key, current_time - 86400)  # 86400 seconds = 24 hours
            )
            result = cursor.fetchone()

            if result:
                logger.debug(f"Using database cached results for '{keyword}'")
                movies = json.loads(result[0])
                search_cache[cache_key] = movies
                return movies
    except Exception as e:
        logger.error(f"Error checking search cache: {e}")

    # If not cached, make the API call
    params = {
        'apikey': '38cf4d5e',
        's': keyword,
        'type': 'movie'
    }
    if year:
        params['y'] = str(year)

    try:
        # Reduced timeout for faster response
        response = requests.get("http://www.omdbapi.com/", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get('Response') == 'True':
            movies = data.get('Search', [])
            # Filter movies from 2000 onwards
            filtered_movies = [m for m in movies if m.get('Year') and 
                              int(m['Year'].split('-')[0]) >= 2000]

            # Cache the results
            search_cache[cache_key] = filtered_movies

            # Store in database
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    current_time = int(datetime.now().timestamp())
                    conn.execute(
                        "INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
                        (cache_key, json.dumps(filtered_movies), current_time)
                    )
            except Exception as e:
                logger.error(f"Error caching search results: {e}")

            return filtered_movies
        else:
            logger.error(f"OMDB API error for '{keyword}': {data.get('Error')}")
            return []

    except requests.RequestException as e:
        logger.error(f"Network/API error for '{keyword}': {str(e)}")
        return []
    except (ValueError, KeyError) as e:
        logger.error(f"Data processing error for '{keyword}': {str(e)}")
        return []

if __name__ == '__main__':
    init_db()  # Initialize the database before starting the app
    app.run(host='0.0.0.0', port=8000, debug=True)
