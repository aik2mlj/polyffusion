    for note in range(12):
    thickness_array=(chroma_matrix>0).sum(axis=1)
    result=np.argwhere(piano_roll>0)[:,1]
    thickness=np.array([get_channel_thickness(ins.get_piano_roll().T) for ins in midi.instruments if not is_percussive_channel(ins)])
