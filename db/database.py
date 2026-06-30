from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

# DATABASE URL:
# We use a local SQLite database named 'docmind.db'.
# Why SQLite:
# 1. Zero Setup: It is file-based and requires no database server installations (unlike PostgreSQL or MySQL).
# 2. Portability: The entire database resides in a single local file, making it easy to share, back up, or delete.
# 3. Perf for single user: Perfect for developer sandboxes, single-user apps, or portfolio demos.
SQLALCHEMY_DATABASE_URL = "sqlite:///docmind.db"

# Create the SQLAlchemy engine.
# connect_args={"check_same_thread": False} is required ONLY for SQLite.
# By default, SQLite only allows one thread to access it at a time. FastAPI can make queries on multiple threads,
# so we disable this check to allow concurrent requests while ensuring safety.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a sessionmaker class which acts as a session factory.
# autocommit=False ensures transactions must be explicitly committed (safer).
# autoflush=False prevents automatic flushing of pending data changes before querying.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initializes the database by creating all defined schema tables.
    If the tables already exist, this call does nothing.
    """
    print("Initializing SQLite database and tables...")
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Generator function that yields a database session and ensures it gets closed.
    
    A DATABASE SESSION represents a temporary transactional workspace for talking to the database.
    It tracks changes to objects and manages the lifecycle of database connections.
    
    This function uses a generator pattern:
    1. It instantiates a session using SessionLocal().
    2. It yields the session to the calling function (typically a FastAPI route via Depends).
    3. Once the route finishes executing, the execution returns here, and the finally block closes the connection.
       This prevents database connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
