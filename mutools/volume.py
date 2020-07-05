from mu.mel import mel


def calc_harmonicity(pitch):
    har = pitch.harmonicity_barlow
    if har == float("inf"):
        return 1
    else:
        return abs(har)


def calc_harmonicity_of_event(poly, polyidx, itemidx,
                              function=calc_harmonicity):
    item = poly[polyidx][itemidx]
    if not isinstance(item.pitch, mel.EmptyPitch):
        simultan = poly.find_exact_simultan_events(polyidx, itemidx)
        h = []
        for event in simultan:
            if not isinstance(event.pitch, mel.EmptyPitch):
                interval = event.pitch - item.pitch
                adapted = function(interval) * (event.duration / item.duration)
                h.append(adapted)
        return sum(h) / len(h)
    else:
        return None


def calc_harmonicity_of_all_events(poly, polyidx,
                                   function=calc_harmonicity):
    return tuple(calc_harmonicity_of_event(poly, polyidx, i, function)
                 for i in range(len(poly[polyidx])))


def calc_volume_by_harmonicity(ls, min_amp, max_amp):
    """
    expect barlow - harmonicity
    (not harmonic complexity like in the euler formula)
    """
    diff_amp = max_amp - min_amp
    maxima = max(ls)
    minima = min(ls)
    diff = maxima - minima
    new = []
    for har in ls:
        comp = (har - minima) / diff
        amp = diff_amp * comp
        amp += min_amp
        new.append(amp)
    return new
