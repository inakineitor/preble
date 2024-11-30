import uuid


def random_uuid_string():
    return str(uuid.uuid4().hex)
