"""Outline and resolution controls for the two headline runs (in the 15-25% band, reference proportions).   python anatomy_feature_controls.py"""
import anatomy_feature_search as S
HEAD = {'planarian': dict(shape='planarian', rows=32, aspect=4.6, range=22.5), 'tadpole': dict(shape='tadpole', rows=24, aspect=4.5, range=25, tail=0.4)}
if __name__ == '__main__':
    cfg = []
    for sh, base in HEAD.items():
        for o in ('planarian', 'tadpole', 'stadium', 'ellipse', 'rectangle'):
            if o != sh:
                cfg.append(dict(base, outline=o))
        cfg.append(dict(base, m=4))
    S.sweep(cfg)
