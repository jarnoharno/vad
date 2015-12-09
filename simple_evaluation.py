def compare_labels(predicted, truth):
    timeline = []
    for i,p in enumerate(predicted):
        if i%2 == 0:
            timeline.append([p, "p", "st"])
        else:
            timeline.append([p, "p", "en"])
    for i,t in enumerate(truth):
        if i%2 == 0:
            timeline.append([t, "t", "st"])
        else:
            timeline.append([t, "t", "en"])
    timeline = sorted(timeline)
    #for i in timeline:
    #    print(i)
    x = 0.0
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    state = 0
    prediction = 0
    for i in timeline:
        if i[1] == "p" and i[2] == "st":
            prediction += 1
            if state > 0:
                fn += i[0]-x
            else:
                tn += i[0]-x
            x = i[0]
        elif i[1] == "p" and i[2] == "en":
            prediction -= 1
            if state > 0:
                tp += i[0]-x
            else:
                fp += i[0]-x
            x = i[0]
        elif i[1] == "t" and i[2] == "st":
            state += 1
            if prediction > 0:
                fp += i[0]-x
            else:
                tn += i[0]-x
            x = i[0]
        elif i[1] == "t" and i[2] == "en":
            state -= 1
            if prediction > 0:
                tp += i[0]-x
            else:
                fn += i[0]-x
            x = i[0]
    return tp, tn, fp, fn
