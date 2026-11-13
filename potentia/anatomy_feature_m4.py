"""Double-resolution check of the 40-cell pinched planarian run (m = 4).   python anatomy_feature_m4.py"""
import anatomy_feature_search as S
if __name__ == '__main__':
    S.sweep([dict(shape='planarian', rows=40, aspect=4.4, range=20, pinch=1, m=4)], procs=1)
