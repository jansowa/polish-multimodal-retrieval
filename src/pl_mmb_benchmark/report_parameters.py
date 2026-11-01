"""Parameter columns for the image/text retrieval paths, in millions."""
HEADERS = ('Enkodery V+T [M]', 'Wizyjny [M]', 'Tekstowy [M]')


def encoder_counts(parameters):
    vision, text = parameters['vision'], parameters['text']
    if any(type(value) is not int or value < 0 for value in (vision, text)):
        raise ValueError('Encoder counts must be nonnegative integers')
    return dict(total=vision + text, vision=vision, text=text)


def parameter_cells(parameters):
    counts = encoder_counts(parameters)
    return ' | '.join(f"{counts[key] / 1e6:.2f}" for key in ('total', 'vision', 'text'))
