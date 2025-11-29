import numpy as np

from api import E, G, L, run

A5 = (148, 210)


def draw(t: float):
    sh_g = G.polygon().label(uid="sphere")
    sh_p = E.affine().fill().label(uid="sphere_eff")
    sp_l = L(sh_p(sh_g))

    v_text_g = G.text().label(uid="v_text")
    v_text_p = E.affine(angles_rad=(0, 0, -np.pi / 2)).fill().label(uid="v_text_eff")
    v_text_l = L(v_text_p(v_text_g))

    h_text_g = G.text().label(uid="h_text")
    h_text_p = E.affine().fill().label(uid="h_text_eff")
    h_text_l = L(h_text_p(h_text_g))

    v_line = G.line().label(uid="v_line")
    v_line_p = E.affine().label(uid="v_line_eff")
    v_line_l = L(v_line_p(v_line))

    return sp_l, v_text_l, h_text_l, v_line_l


if __name__ == "__main__":
    run(draw, canvas_size="A5", render_scale=6, show_hud=False)
