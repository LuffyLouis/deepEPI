
def left_pad_string(s, target_length, fill_char='N'):
    return s.rjust(target_length, fill_char)

def right_pad_string(s, target_length, fill_char='N'):
    return s.ljust(target_length, fill_char)

def center_pad_string(s, target_length, fill_char='N'):
    return s.center(target_length, fill_char)

def trim_string_to_length(s, target_length, side='right'):
    if len(s) <= target_length:
        return s
    if side == 'left':
        return s[:target_length]
    elif side == 'right':
        return s[-target_length:]
    else:  # 'both' or 'center'
        return s[:target_length] if len(s) % 2 == 0 else s[-target_length:]

def pad_string_to_length(s, target_length, fill_char='N', side='right'):
    if len(s) >= target_length:
        return s
    if side == 'left':
        return left_pad_string(s, target_length, fill_char)
    elif side == 'right':
        return right_pad_string(s, target_length, fill_char)
    else:  # 'both' or 'center'
        return center_pad_string(s, target_length, fill_char)


def generate_k_mers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    # for i in range(len(sequence) - k + 1):
    #     yield sequence[i:i + k]