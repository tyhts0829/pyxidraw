#!/usr/bin/env python3

import math
import numbers
import operator


####
# 直径1の球に内接する正多面体のデータ生成
####


class RegularPolyhedron:
    INDEX = {4: 1, 6: 2, 8: 3, 12: 4, 20: 5}
    POLYHEDRON = (4, 4, 6, 8, 12, 20)
    POLYGON = (3, 3, 4, 3, 5, 3)

    # ３ベクトルから残りのベクトルを求める.
    @staticmethod
    def create_vector_list(p1, p2, p3):
        v1 = p1.normalize()
        v2 = (p1 * p2).normalize()
        v3 = (v1 * v2).normalize()
        xcos, ycos, zcos = [v.dot(p3) for v in (v1, v2, v3)]
        pcos = p1.dot(p2)
        vecs = [p1, p2, p3]

        def vec_add(q1, q2):
            x = q1.normalize()
            y = (q1 * q2).normalize()
            z = (x * y).normalize()
            v = (x * xcos + y * ycos + z * zcos).normalize()
            if len(tuple(filter((lambda p: v.neareq(p)), vecs))) == 0:
                vecs.append(v)

        last_vlen = 0
        new_vlen = len(vecs)
        while last_vlen != new_vlen:
            for q1 in vecs[last_vlen:new_vlen]:
                for q2 in vecs:
                    if q1.doteq(q2, pcos):
                        vec_add(q1, q2)
                        vec_add(q2, q1)
            last_vlen = new_vlen
            new_vlen = len(vecs)

        return tuple(
            [v.fixfp() for v in sorted(vecs, key=operator.itemgetter(2, 1, 0), reverse=True)]
        )

    # ポリゴン情報の一覧を作成.
    @staticmethod
    def create_polygon_list(vertex, normal, pcos, ncos):
        vlen = len(vertex)
        r = []
        for n in normal:
            pvl = tuple(filter((lambda vi: n.doteq(vertex[vi], ncos)), range(vlen)))
            pvlen = len(pvl)
            ll = []
            for ci in range(pvlen):
                pc = vertex[pvl[ci]]
                pi, ni = tuple(filter(lambda qi: pc.doteq(vertex[pvl[qi]], pcos), range(pvlen)))
                pp, pn = vertex[pvl[pi]], vertex[pvl[ni]]
                pi, ni = (pi, ni) if n.neareq(((pc - pp) * (pn - pc)).normalize()) else (ni, pi)
                ll.append([ci, ni])
            ci, ni = ll[0]
            vl = [pvl[ci]]
            while ni != 0:
                vl.append(pvl[ni])
                ci, ni = ll[ni]
            r.append(tuple(vl))
        return tuple(r)

    # 辺情報の一覧を作成.
    @staticmethod
    def create_edge_list(polygon):
        r = []
        for poly in polygon:
            pp = poly[-1]
            for pn in poly:
                e = (pp, pn) if pp < pn else (pn, pp)
                if e not in r:
                    r.append(e)
                pp = pn
        return tuple(sorted(r))

    # コンストラクタ.
    def __init__(self, M):
        S = RegularPolyhedron
        self.M = M
        self.I = S.INDEX[M]
        self.V = S.POLYHEDRON[self.I ^ 1]
        self.F = S.POLYGON[self.I ^ 1]
        self.N = S.POLYGON[self.I]

        PPF = math.pi / self.F
        PPN = math.pi / self.N

        # 最初の３頂点を求める.
        rotXY = 2 * PPF
        cosXY = math.cos(rotXY)
        sinXY = math.sin(rotXY)
        cosZX = 2 * math.pow(math.cos(PPN) / math.sin(PPF), 2) - 1
        sinZX = math.sqrt(1 - cosZX * cosZX)

        p1 = Vector3(0, 0, 1)
        p2 = Vector3(sinZX, 0, cosZX)
        p3 = Vector3(sinZX * cosXY, sinZX * sinXY, cosZX)

        # 最初の３法線を求める.
        rm = Matrix33(cosXY, -sinXY, 0, sinXY, cosXY, 0, +0, 0, 1)
        n1 = ((p1 - p3) * (p2 - p1)).normalize()
        n2 = (rm * n1).normalize()
        n3 = (rm * n2).normalize()

        # 頂点と法線の一覧を作成する.
        self.main_vertex = S.create_vector_list(p1, p2, p3)
        self.main_normal = S.create_vector_list(n1, n2, n3)
        self.dual_vertex = self.main_normal
        self.dual_normal = self.main_vertex
        self.vertex = self.main_vertex
        self.normal = self.main_normal

        # 正多面体と対の両方の多角形データを作成する.
        self.main_polygon = S.create_polygon_list(
            self.main_vertex,
            self.main_normal,
            math.cos(2 * PPN),
            math.cos(2 * PPF),
        )
        self.dual_polygon = S.create_polygon_list(
            self.dual_vertex,
            self.dual_normal,
            math.cos(2 * PPF),
            math.cos(2 * PPN),
        )
        self.polygon = self.main_polygon

        # 正多面体と対の両方の辺データを作成する.
        self.main_edge = S.create_edge_list(self.main_polygon)
        self.dual_edge = S.create_edge_list(self.dual_polygon)
        self.edge = self.main_edge


####
# Vector3
####


class Vector3(numbers.Integral):
    MESSAGE_FORMAT = "(%+.6e, %+.6e, %+.6e)"

    def __init__(self, *args):
        if len(args) == 3:
            self.value = list(args)
            return
        if len(args) == 0:
            self.value = [0, 0, 0]
            return
        if len(args) == 1:
            self.value = args[0]
            return
        raise TypeError

    def __int__(self):
        raise NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return Vector3.MESSAGE_FORMAT % tuple(self.value)

    def __bool__(self):
        return sum(self.value) != 0

    def __eq__(self, rhs):
        if type(rhs) is Vector3:
            return self.value == rhs.value
        return False

    def __lt__(self, rhs):
        raise NotImplemented

    def __le__(self, rhs):
        raise NotImplemented

    def __neg__(self):
        x, y, z = self.value
        return Vector3(-x, -y, -z)

    def __pos__(self):
        return Vector3(self)

    def __invert__(self):
        raise NotImplemented

    def __add__(self, rhs):
        if type(rhs) is Vector3:
            x1, y1, z1 = self.value
            x2, y2, z2 = rhs.value
            return Vector3(x1 + x2, y1 + y2, z1 + z2)
        raise NotImplemented

    def __sub__(self, rhs):
        if type(rhs) is Vector3:
            x1, y1, z1 = self.value
            x2, y2, z2 = rhs.value
            return Vector3(x1 - x2, y1 - y2, z1 - z2)
        raise NotImplemented

    def __mul__(self, rhs):
        x1, y1, z1 = self.value
        if type(rhs) in (int, float):
            return Vector3(x1 * rhs, y1 * rhs, z1 * rhs)
        if type(rhs) is Vector3:
            x2, y2, z2 = rhs.value
            return Vector3(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)
        if type(rhs) is Matrix33:
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = rhs.value
            return Vector3(
                x1 * m11 + y1 * m21 + z1 * m31,
                x1 * m12 + y1 * m22 + z1 * m32,
                x1 * m13 + y1 * m23 + z1 * m33,
            )
        raise NotImplemented

    def __pow__(self, rhs):
        raise NotImplemented

    def __divmod__(self, rhs):
        if type(rhs) in (int, float):
            return [self.__truediv__(rhs), self.__mod__(rhs)]
        raise NotImplemented

    def __truediv__(self, rhs):
        if type(rhs) in (int, float):
            x, y, z = self.value
            return Vector3(x / rhs, y / rhs, z / rhs)
        raise NotImplemented

    def __floordiv__(self, rhs):
        if type(rhs) in (int, float):
            x, y, z = self.value
            return Vector3(math.floor(x / rhs), math.floor(y / rhs), math.floor(z / rhs))
        raise NotImplemented

    def __mod__(self, rhs):
        if type(rhs) in (int, float):
            x, y, z = self.value
            return Vector3(x % rhs, y % rhs, z % rhs)
        raise NotImplemented

    # 内積とする.
    def __and__(self, rhs):
        if type(rhs) is Vector3:
            return self.dot(rhs)
        raise NotImplemented

    def __xor__(self, rhs):
        raise NotImplemented

    def __or__(self, rhs):
        raise NotImplemented

    def __lshift__(self, rhs):
        raise NotImplemented

    def __rshift__(self, rhs):
        raise NotImplemented

    def __radd__(self, lhs):
        raise NotImplemented

    def __rsub__(self, lhs):
        raise NotImplemented

    def __rmul__(self, lhs):
        raise NotImplemented

    def __rmod__(self, lhs):
        raise NotImplemented

    def __rtruediv__(self, lhs):
        raise NotImplemented

    def __rfloordiv__(self, lhs):
        raise NotImplemented

    def __rpow__(self, lhs):
        raise NotImplemented

    def __rand__(self, lhs):
        raise NotImplemented

    def __rxor__(self, lhs):
        raise NotImplemented

    def __ror__(self, lhs):
        raise NotImplemented

    def __rlshift__(self, lhs):
        raise NotImplemented

    def __rrshift__(self, lhs):
        raise NotImplemented

    def __iadd__(self, rhs):
        if type(rhs) is Vector3:
            x1, y1, z1 = self.value
            x2, y2, z2 = rhs.value
            self.value = [x1 + x2, y1 + y2, z1 + z2]
            return self
        raise NotImplemented

    def __isub__(self, rhs):
        if type(rhs) is Vector3:
            x1, y1, z1 = self.value
            x2, y2, z2 = rhs.value
            self.value = [x1 - x2, y1 - y2, z1 - z2]
            return self
        raise NotImplemented

    def __imul__(self, rhs):
        x1, y1, z1 = self.value
        if type(rhs) in (int, float):
            self.value = (x1 * rhs, y1 * rhs, z1 * rhs)
            return self
        if type(rhs) is Vector3:
            x2, y2, z2 = rhs.value
            self.value = [y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2]
            return self
        if type(rhs) is Matrix33:
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = rhs.value
            self.value = [
                x1 * m11 + y1 * m21 + z1 * m31,
                x1 * m12 + y1 * m22 + z1 * m32,
                x1 * m13 + y1 * m23 + z1 * m33,
            ]
            return self
        raise NotImplemented

    def __itruediv__(self, rhs):
        if type(rhs) in (int, float):
            x, y, z = self.value
            self.value = [x / rhs, y / rhs, z / rhs]
            return self
        raise NotImplemented

    def __imod__(self, rhs):
        if type(rhs) in (int, float):
            x, y, z = self.value
            self.value = [x % rhs, y % rhs, z % rhs]
            return self
        raise NotImplemented

    def __abs__(self):
        x, y, z = self.value
        return math.sqrt(x * x + y * y + z * z)

    def __ceil__(self):
        x, y, z = self.value
        return Vector3(math.ceil(x), math.ceil(y), math.ceil(z))

    def __floor__(self):
        x, y, z = self.value
        return Vector3(math.floor(x), math.floor(y), math.floor(z))

    def __round__(self):
        x, y, z = self.value
        return Vector3(round(x), round(y), round(z))

    def __trunc__(self):
        x, y, z = self.value
        return Vector3(math.trunc(x), math.trunc(y), math.trunc(z))

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    ####

    def dot(self, rhs):
        x1, y1, z1 = self.value
        x2, y2, z2 = rhs.value
        return x1 * x2 + y1 * y2 + z1 * z2

    def neareq(self, rhs, thr=(1 / (1 << 20))):
        x1, y1, z1 = self.value
        x2, y2, z2 = rhs.value
        return math.fabs(x1 - x2) < thr and math.fabs(y1 - y2) < thr and math.fabs(z1 - z2) < thr

    def doteq(self, rhs, thr):
        return math.fabs(self.dot(rhs) - thr) <= 1 / (1 << 20)

    def normalize(self):
        x, y, z = self.value
        l = math.sqrt(x * x + y * y + z * z)
        return Vector3(x / l, y / l, z / l)

    def fixfp(self):
        x, y, z = [int(v * (1 << 20)) / (1 << 20) for v in self.value]
        return Vector3(x, y, z)


####
# Matrix33
####


class Matrix33(numbers.Integral):
    def __init__(self, *args):
        if len(args) == 9:
            self.value = list(args)
            return
        if len(args) == 0:
            self.value = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            return
        if len(args) == 1:
            self.value = args[0]
            return
        raise TypeError

    def __int__(self):
        raise NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
        return "[[%.6e, %.6e, %.6e], [%.6e, %.6e, %.6e], [%.6e, %.6e, %.6e]]" % (
            m11,
            m12,
            m13,
            m21,
            m22,
            m23,
            m31,
            m32,
            m33,
        )

    def __bool__(self):
        m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
        return (
            (m11 != 0)
            or (m12 != 0)
            or (m13 != 0)
            or (m21 != 0)
            or (m22 != 0)
            or (m23 != 0)
            or (m31 != 0)
            or (m32 != 0)
            or (m33 != 0)
        )

    def __eq__(self, rhs):
        if type(rhs) is Matrix33:
            return self.value == rhs.value
        return False

    def __lt__(self, rhs):
        raise NotImplemented

    def __le__(self, rhs):
        raise NotImplemented

    def __neg__(self):
        m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
        return Matrix33(-m11, -m12, -m13, -m21, -m22, -m23, -m31, -m32, -m33)

    def __pos__(self):
        return Matrix33(self)

    def __invert__(self):
        raise NotImplemented

    def __add__(self, rhs):
        if type(rhs) is Matrix33:
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
            r11, r12, r13, r21, r22, r23, r31, r32, r33 = rhs.value
            return Matrix33(
                m11 + r11,
                m12 + r12,
                m13 + r13,
                m21 + r21,
                m22 + r22,
                m23 + r23,
                m31 + r31,
                m32 + r32,
                m33 + r33,
            )
        raise NotImplemented

    def __sub__(self, rhs):
        if type(rhs) is Matrix33:
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
            r11, r12, r13, r21, r22, r23, r31, r32, r33 = rhs.value
            return Matrix33(
                m11 - r11,
                m12 - r12,
                m13 - r13,
                m21 - r21,
                m22 - r22,
                m23 - r23,
                m31 - r31,
                m32 - r32,
                m33 - r33,
            )
        raise NotImplemented

    def __mul__(self, rhs):
        m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
        if type(rhs) in (int, float):
            return Matrix33(
                m11 * rhs,
                m12 * rhs,
                m13 * rhs,
                m21 * rhs,
                m22 * rhs,
                m23 * rhs,
                m31 * rhs,
                m32 * rhs,
                m33 * rhs,
            )
        if type(rhs) is Vector3:
            x, y, z = rhs.value
            return Vector3(
                m11 * x + m12 * y + m13 * z,
                m21 * x + m22 * y + m23 * z,
                m31 * x + m32 * y + m33 * z,
            )
        if type(rhs) is Matrix33:
            r11, r12, r13, r21, r22, r23, r31, r32, r33 = rhs.value
            return Matrix33(
                m11 * r11 + m12 * r21 + m13 * r31,
                m21 * r11 + m22 * r21 + m23 * r31,
                m31 * r11 + m32 * r21 + m33 * r31,
                m11 * r12 + m12 * r22 + m13 * r32,
                m21 * r12 + m22 * r22 + m23 * r32,
                m31 * r12 + m32 * r22 + m33 * r32,
                m11 * r13 + m12 * r23 + m13 * r33,
                m21 * r13 + m22 * r23 + m23 * r33,
                m31 * r13 + m32 * r23 + m33 * r33,
            )
        raise NotImplemented

    def __pow__(self, rhs):
        raise NotImplemented

    def __divmod__(self, rhs):
        if type(rhs) in (int, float):
            return [self.__truediv__(rhs), self.__mod__(rhs)]
        raise NotImplemented

    def __truediv__(self, rhs):
        if type(rhs) in (int, float):
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
            return Matrix33(
                m11 / rhs,
                m12 / rhs,
                m13 / rhs,
                m21 / rhs,
                m22 / rhs,
                m23 / rhs,
                m31 / rhs,
                m32 / rhs,
                m33 / rhs,
            )
        raise NotImplemented

    def __floordiv__(self, rhs):
        if type(rhs) in (int, float):
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
            return Matrix33(
                math.floor(m11 / rhs),
                math.floor(m12 / rhs),
                math.floor(m13 / rhs),
                math.floor(m21 / rhs),
                math.floor(m22 / rhs),
                math.floor(m23 / rhs),
                math.floor(m31 / rhs),
                math.floor(m32 / rhs),
                math.floor(m33 / rhs),
            )
        raise NotImplemented

    def __mod__(self, rhs):
        if type(rhs) in (int, float):
            m11, m12, m13, m21, m22, m23, m31, m32, m33 = self.value
            return Matrix33(
                m11 % rhs,
                m12 % rhs,
                m13 % rhs,
                m21 % rhs,
                m22 % rhs,
                m23 % rhs,
                m31 % rhs,
                m32 % rhs,
                m33 % rhs,
            )
        raise NotImplemented

    def __and__(self, rhs):
        raise NotImplemented

    def __xor__(self, rhs):
        raise NotImplemented

    def __or__(self, rhs):
        raise NotImplemented

    def __lshift__(self, rhs):
        raise NotImplemented

    def __rshift__(self, rhs):
        raise NotImplemented

    def __radd__(self, lhs):
        raise NotImplemented

    def __rsub__(self, lhs):
        raise NotImplemented

    def __rmul__(self, lhs):
        raise NotImplemented

    def __rmod__(self, lhs):
        raise NotImplemented

    def __rtruediv__(self, lhs):
        raise NotImplemented

    def __rfloordiv__(self, lhs):
        raise NotImplemented

    def __rpow__(self, lhs):
        raise NotImplemented
