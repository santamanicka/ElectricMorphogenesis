"""Outline and resolution controls for the two headline runs (in the 15-25% band, reference proportions).   python anatomy_feature_controls.py"""
import anatomy_feature_search as S
HEAD = {'planarian': dict(shape='planarian', rows=40, aspect=4.4, range=20, pinch=1), 'tadpole': dict(shape='tadpole', rows=24, aspect=4.5, range=25, tail=0.4)}
if __name__ == '__main__':
    cfg = []
    for sh, base in HEAD.items():
        if sh == 'tadpole':
            continue                                         # done already (rows 24, range 25)
        for o in ('planarian', 'tadpole', 'stadium', 'ellipse', 'rectangle'):
            if o != sh:
                cfg.append(dict(base, outline=o))
        if sh == 'tadpole':
            cfg.append(dict(base, m=4))
    S.sweep(cfg)
