from sqlalchemy import Column, Integer, String, Text, Date, Boolean, TIMESTAMP, ForeignKey, BigInteger
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), nullable=False)
    user_type = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)

class UserDetails(Base):
    __tablename__ = "userdetails"
    id = Column(Integer, primary_key=True)
    bio = Column(Text)
    avatar_path = Column(Text)
    birth_date = Column(Date)
    phone = Column(String(20))
    city = Column(String(100))
    subscription_active = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP, nullable=False)

class AdminDetails(Base):
    __tablename__ = "admindetails"
    id = Column(Integer, primary_key=True)
    department = Column(String(100))
    phone = Column(String(20))
    permissions_level = Column(Integer, nullable=False, default=1)
    access_code = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)

class Tasks(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(255), nullable=False, default="pending")
    created_at = Column(TIMESTAMP, nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    finished_at = Column(TIMESTAMP, nullable=True)
    result_path = Column(Text, nullable=True)
    description = Column(Text, nullable=True)

    user = relationship("Users", backref="tasks")

class Files(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(255), nullable=False)
    storage_path = Column(Text, nullable=False)
    uploaded_at = Column(TIMESTAMP, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)

    user = relationship("Users", backref="files")
    task = relationship("Tasks", backref="files")

class Nodes(Base):
    __tablename__ = "nodes"
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(255), nullable=False)
    device_type = Column(String(255), nullable=False)
    status = Column(String(255), nullable=False)

class TasksNodes(Base):
    __tablename__ = "tasks_nodes"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    assigned_at = Column(TIMESTAMP, nullable=False)
    status = Column(String(255), nullable=False, default="assigned")

    task = relationship("Tasks", backref="tasks_nodes")
    node = relationship("Nodes", backref="tasks_nodes")

class TasksLogs(Base):
    __tablename__ = "tasks_logs"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    log_level = Column(String(255), nullable=False, default="info")
    message = Column(Text, nullable=False)

    task = relationship("Tasks", backref="logs")
    node = relationship("Nodes", backref="logs")

# Backwards-compatibility aliases (some modules expect singular class names)
Task = Tasks
File = Files
Node = Nodes
TasksNode = TasksNodes
TasksLog = TasksLogs
