

def get_name(object_):
    try:
        return object_.name
    except AttributeError:
        return object_.__name__
