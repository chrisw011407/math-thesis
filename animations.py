import numpy as np
from manim import *

# # # MODULE ON VECTORS! # # #


# Locating a friend.


class LocateFriend(Scene):
    # Construct a grid with two nodes, one labeled "You" and the other labeled "Friend".
    def construct(self):
        # Create a grid with two nodes.
        grid = NumberPlane(
            x_range=[0, 6, 1],  # x-axis from 0 to 6 with step size 1
            y_range=[0, 6, 1],  # y-axis from 0 to 6 with step size 1
            axis_config={"include_numbers": True}
        ).shift(LEFT * 2)
        # Create a text box to the right of the grid.
        text = Tex(r"Suppose you have \\ just located your friend.").next_to(
            grid, direction=RIGHT)
        youNode = Dot().move_to(np.array([-2, -1, 0]))
        friendNode = Dot().move_to(np.array([-2, 2, 0]))
        you = Tex(r"You").move_to(
            youNode.get_center() + np.array([0.6, -0.4, 0]))
        friend = Tex(r"Friend").move_to(
            friendNode.get_center() + np.array([0.8, -0.4, 0]))
        # Draw vector from "You" to "Friend".
        vec = Arrow(start=friendNode.get_center(),
                    end=youNode.get_center(), color=BLUE)
        vecLabel = MathTex(r"\mathbf{s}", color=BLUE).next_to(
            vec, direction=LEFT)
        text2 = Tex(r"Now, you walk towards \\ your friend.").next_to(
            grid, direction=RIGHT)
        # Move "You" node towards "Friend" node.
        aVec = Arrow(start=np.array([-2, -1.01, 0]),
                     end=youNode.get_center(), color=RED)
        aVeclabel = MathTex(r"\mathbf{a}", color=RED).next_to(
            aVec, direction=LEFT)

        def update_vec(mob):
            mob.become(Arrow(start=friendNode.get_center(),
                             end=youNode.get_center(), color=BLUE, max_tip_length_to_length_ratio=0.2))

        def update_avec(mob):
            mob.become(Arrow(start=np.array([-2, -1.01, 0]),
                             end=youNode.get_center(), color=RED, max_tip_length_to_length_ratio=0.2))

        vec.add_updater(update_vec)
        vecLabel.add_updater(lambda x: x.next_to(vec, direction=LEFT))
        aVec.add_updater(update_avec)
        aVeclabel.add_updater(lambda x: x.next_to(aVec, direction=LEFT))
        self.play(FadeIn(grid), Write(text), Create(youNode), Create(friendNode),
                  Write(you), Write(friend), run_time=2)
        self.wait(1)
        self.play(Create(vec), Write(vecLabel), run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(text, text2), run_time=1)
        self.add(aVec, aVeclabel)
        self.play(Create(aVec), Write(aVeclabel), run_time=2)
        self.wait(1)
        self.play(youNode.animate.move_to(
            friendNode.get_center()), run_time=10)

# Matrix multiplication on vector.


class MatrixOnVector(Scene):
    def construct(self):
        # Create a matrix and a vector.
        capt = Tex(
            r"Suppose we have a vector \\ $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.", font_size=36).shift(UP*3, LEFT*3)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 2, 1],
            axis_config={"include_numbers": True}
        )
        vec = Arrow(start=axes.c2p(0, 0), end=axes.c2p(1, 1), color=RED)
        vecLabel = Matrix([[1], [1]]).set_color(
            RED).next_to(vec, direction=RIGHT)
        capt2 = Tex(r"Let's apply the matrix \\[0.9em] $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ to the vector.", font_size=36).shift(
            UP*3, LEFT*3)
        capt3 = Tex(r"To get the first entry of the resulting vector,\\ we multiply the first row of the matrix by \\ the vector coordinate-wise and add.",
                    font_size=30).shift(UP*3, LEFT*3)
        vecLabel2 = vecLabel.copy().shift(UP*3, RIGHT*1)
        matrix = Matrix([[0, -1], [1, 0]]
                        ).set_color(BLUE).next_to(vecLabel2, direction=LEFT)
        m1 = matrix.copy()
        m1.add(SurroundingRectangle(m1.get_rows()[0]))
        m2 = matrix.copy()
        m2.add(SurroundingRectangle(m2.get_rows()[1]).set_color(WHITE))
        vL3 = vecLabel2.copy()
        vL3.add(SurroundingRectangle(vL3.get_columns()[0]))
        vL4 = vecLabel2.copy()
        vL4.add(SurroundingRectangle(vL4.get_columns()[0]).set_color(WHITE))
        result = Matrix([["x_1"], ["x_2"]]).set_color(
            GREEN).next_to(vL3, direction=DOWN)
        equals = MathTex(r"=").next_to(result, direction=LEFT)
        result1 = Matrix([["0\cdot 1 + -1\cdot 1"], ["x_2"]]).set_color(
            GREEN).next_to(vL3, direction=DOWN)
        equals1 = equals.copy().next_to(result1, direction=LEFT)
        result2 = Matrix([["-1"], ["1\cdot 1 + 0\cdot 1"]]).set_color(
            GREEN).next_to(vL3, direction=DOWN)
        equals2 = equals.copy().next_to(result2, direction=LEFT)
        result3 = Matrix([["-1"], ["1"]]).set_color(
            GREEN).next_to(vL3, direction=DOWN)
        equals3 = equals.copy().next_to(result3, direction=LEFT)
        ResultVec = Arrow(start=axes.c2p(
            0, 0), end=axes.c2p(-1, 1), color=GREEN)
        resultVecLabel = result3.copy().next_to(ResultVec, direction=LEFT)
        self.add(capt, axes)
        self.play(Create(vec), run_time=1)
        self.wait(0.5)
        self.play(Write(vecLabel), run_time=1)
        self.wait(1)
        self.play(ReplacementTransform(capt, capt2), ReplacementTransform(
            vecLabel, vecLabel2), Write(matrix), run_time=1)
        self.wait(3)
        self.play(ReplacementTransform(capt2, capt3), run_time=1)
        self.wait(1)
        self.play(Create(m1), Create(vL3), Write(
            equals), Create(result), run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(m1.get_rows()[
                  0], vL3.get_columns()[0]), run_time=1)
        self.wait(0.5)
        self.play(ReplacementTransform(
            result, result1), ReplacementTransform(equals, equals1), run_time=3)
        self.wait(1)
        self.play(ReplacementTransform(m1, m2), ReplacementTransform(
            vL3, vL4), run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(m2.get_rows()[
                  1], vL4.get_columns()[0]), run_time=1)
        self.wait(0.5)
        self.play(ReplacementTransform(result1, result2),
                  ReplacementTransform(equals1, equals2), run_time=3)
        self.wait(1)
        self.play(ReplacementTransform(result2, result3),
                  ReplacementTransform(equals2, equals3), run_time=2)
        self.wait(2)
        self.play(FadeOut(equals3), ReplacementTransform(vec, ResultVec), ReplacementTransform(
            result3, resultVecLabel), run_time=3)
        self.wait()

# Matrix multiplication and composition of linear transformations.


class MatrixMult(Scene):
    def construct(self):
        # Create axes.
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_numbers": True}, tips=False, x_length=7, y_length=7
        ).shift(LEFT*3)
        capt1 = Tex(
            r"Here are a couple examples of matrix multiplication. \\ First, let's start with square matrices of dimension 3.", font_size=48)
        mat1 = MathTex(r"\text{Let } A = \begin{bmatrix} 1 & 4 & 5 \\ 0 & 1 & 4 \\ 3 & 3 & 5 \end{bmatrix}").shift(
            LEFT*3).set_color(ORANGE)
        mat2 = MathTex(r"\text{and } B = \begin{bmatrix} 1 & 1 & 0 \\ 3 & 2 & 0 \\ 2 & 3 & 1 \end{bmatrix}").next_to(
            mat1, direction=RIGHT, buff=0.5).set_color(BLUE)
        m1 = Matrix([[1, 4, 5], [0, 1, 4], [3, 3, 5]]).set_color(ORANGE)
        m2 = Matrix([[1, 1, 0], [3, 2, 0], [2, 3, 1]]).set_color(
            BLUE).next_to(m1, direction=RIGHT)

        self.add(mat1, mat2)
        # Create a vector to transform


class Wave(Scene):
    def construct(self):
        eqn = MathTex(r"w(t,x) = A\sin(kx - \omega t)",
                      substrings_to_isolate=["A", "k", "\omega"]).shift(UP*3, LEFT*4)
        eqn.set_color_by_tex("A", BLUE)
        eqn.set_color_by_tex("k", RED)
        eqn.set_color_by_tex("\omega", GREEN)
        ax_1 = Axes(
            x_range=[-5, 5, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_numbers": True},
            tips=False
        )

        def wave1(x):
            return np.sin(x)

        def wave2(x):
            return 0.25*np.sin(x)

        def wave3(x):
            return np.sin(2*x)

        def A_update_func(t):
            return -0.75 * smooth(t)

        def A_update_func2(t):
            return 0.75 * smooth(t)

        wave1_graph = ax_1.plot(wave1, x_range=(-5, 5, 0.01), color=WHITE)
        wave2_graph = ax_1.plot(wave2, x_range=(-5, 5, 0.01), color=BLUE)
        wave_3_graph = ax_1.plot(wave3, x_range=(-5, 5, 0.01), color=RED)
        dot1 = Dot().move_to(ax_1.c2p(- PI / 2, -1))
        dot2 = Dot().move_to(ax_1.c2p(3 * PI / 2, -1))
        dot3 = Dot().move_to(ax_1.c2p(PI / 4, 1))
        dot4 = Dot().move_to(ax_1.c2p(5 * PI / 4, 1))
        brace = BraceBetweenPoints(dot1.get_center(), dot2.get_center())
        brace_text = brace.get_tex(r"\lambda_0 = 2\pi / k")
        brace2 = BraceBetweenPoints(dot3.get_center(), dot4.get_center())
        brace_text2 = brace2.get_tex(r"\lambda = \lambda_0 / 2")
        brace_text2.set_color(RED)
        A_label = MathTex("A =", substrings_to_isolate="A").shift(
            DOWN*3, RIGHT*2)
        A_label.set_color_by_tex("A", BLUE)
        A_value = DecimalNumber(1).next_to(A_label, direction=RIGHT)
        A_line = Line(start=ax_1.c2p(PI / 2, 0),
                      end=ax_1.c2p(PI / 2, 1), color=BLUE)
        A_line_label = MathTex("A", color=BLUE).next_to(
            A_line, direction=RIGHT)
        K_label = MathTex("k =", substrings_to_isolate="k").shift(
            DOWN*3, RIGHT*2)
        K_label.set_color_by_tex("k", RED)
        K_value = DecimalNumber(1).next_to(K_label, direction=RIGHT)

        self.play(Create(ax_1), Write(eqn), run_time=3)
        self.wait(1)
        self.play(Create(wave1_graph), Create(dot1), Create(dot2), run_time=1)
        self.wait(1)
        self.play(Create(brace), Write(brace_text), run_time=2)
        self.wait(1)
        self.play(Uncreate(brace), Unwrite(brace_text),
                  Uncreate(dot1), Uncreate(dot2), run_time=1)
        self.wait(1)
        self.play(Write(A_label), Write(A_value), Create(
            A_line), Write(A_line_label), run_time=2)
        self.wait(1)
        self.play(Uncreate(A_line), Unwrite(A_line_label), run_time=1)
        wave1_graph.save_state()
        self.play(ReplacementTransform(wave1_graph, wave2_graph),
                  ChangeDecimalToValue(A_value, 0.25), run_time=4)
        self.wait(1)
        self.remove(wave2_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            A_value, 1), run_time=4)
        self.wait(1)
        self.play(Uncreate(A_label), Uncreate(A_value), run_time=1)
        self.wait(1)
        self.play(Write(K_label), Write(K_value), run_time=1)
        self.wait(1)
        self.play(ReplacementTransform(wave1_graph, wave_3_graph),
                  ChangeDecimalToValue(K_value, 2), run_time=4)
        self.wait(1)
        self.play(Create(dot3), Create(dot4), Create(
            brace2), Write(brace_text2), run_time=2)
        self.wait(1)
        self.play(Uncreate(dot3), Uncreate(dot4), Uncreate(
            brace2), Unwrite(brace_text2), run_time=1)
        self.remove(wave_3_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            K_value, 1), run_time=4)
        self.wait()


class StandingWave(Scene):
    def construct(self):
        ax_1 = Axes(
            x_range=[-2 * PI, 2 * PI, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_numbers": True},
            tips=False
        )

        wavelabel = MathTex(
            r"w(x,t) = 2A\sin(kx)\cos(\omega t)").shift(UP*3, LEFT*3.5)

        def standwave(x):
            return np.sin(x)

        def standwave2(x):
            return -np.sin(x)

        wave1_graph = ax_1.plot(
            standwave, x_range=(-2 * PI, 2 * PI, 0.01), color=WHITE)
        wave2_graph = ax_1.plot(standwave2,
                                x_range=(-2 * PI, 2 * PI, 0.01), color=WHITE)
        omega_label = MathTex(r"\omega =").shift(DOWN*3, LEFT*3)
        omega_value = MathTex(r"\pi").next_to(omega_label, direction=RIGHT)
        omega2_value = MathTex(r"2\pi").next_to(omega_label, direction=RIGHT)
        time_label = MathTex(r"t =").shift(DOWN*3, RIGHT*3)
        time_value = DecimalNumber(0).next_to(time_label, direction=RIGHT)
        caption = Tex(r"Observe how the standing wave's \\ amplitude is modulated by the \\ cosine function.",
                      font_size=30).shift(UP*3, RIGHT*4)
        nodes = [Dot().move_to(ax_1.c2p(-PI, 0)), Dot().move_to(ax_1.c2p(PI, 0)), Dot().move_to(
            ax_1.c2p(0, 0)), Dot().move_to(ax_1.c2p(2*PI, 0)), Dot().move_to(ax_1.c2p(-2*PI, 0))]
        antinodes = [Dot().move_to(ax_1.c2p(-PI/2, -1)), Dot().move_to(ax_1.c2p(PI/2, 1)),
                     Dot().move_to(ax_1.c2p(3*PI/2, -1)), Dot().move_to(ax_1.c2p(-3*PI/2, 1))]
        # make nodes red
        for i, node in enumerate(nodes):
            node.set_color(RED).scale(2)
        # make antinodes green
        for i, antinode in enumerate(antinodes):
            antinode.set_color(GREEN).scale(2)
        node_label = Tex(r"Nodes", color=RED).shift(DOWN*2, LEFT*5)
        antinode_label = Tex(r"Antinodes", color=GREEN).shift(DOWN*2, LEFT*5)
        wave1_graph.save_state()
        time_value.save_state()
        self.play(Create(ax_1), Create(wave1_graph),
                  Write(wavelabel), Write(omega_label), Write(omega_value), Write(time_label), Write(time_value), Write(caption), run_time=3)
        self.wait(2)
        for i, node in enumerate(nodes):
            self.play(Create(node), run_time=0.5)
        self.play(Write(node_label), run_time=1)
        self.wait(1)
        self.play(Unwrite(node_label), run_time=1)
        self.wait(1)
        for i, antinode in enumerate(antinodes):
            self.play(Create(antinode), run_time=0.5)
        self.play(Write(antinode_label), run_time=1)
        self.wait(1)
        for i, antinode in enumerate(antinodes):
            self.play(Uncreate(antinode), run_time=0.5)
        self.play(Unwrite(antinode_label), run_time=1)
        self.wait(1)
        self.play(ReplacementTransform(wave1_graph, wave2_graph),
                  ChangeDecimalToValue(time_value, 1), run_time=1)
        self.remove(wave2_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            time_value, 2), run_time=1)
        self.play(ReplacementTransform(wave1_graph, wave2_graph),
                  ChangeDecimalToValue(time_value, 3), run_time=1)
        self.remove(wave2_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            time_value, 4), run_time=1)
        self.wait(2)
        self.play(ReplacementTransform(omega_value, omega2_value),
                  Restore(time_value), run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(wave1_graph, wave2_graph),
                  ChangeDecimalToValue(time_value, 0.5), run_time=0.5)
        self.remove(wave2_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            time_value, 1), run_time=0.5)
        self.play(ReplacementTransform(wave1_graph, wave2_graph),
                  ChangeDecimalToValue(time_value, 1.5), run_time=0.5)
        self.remove(wave2_graph)
        self.play(Restore(wave1_graph), ChangeDecimalToValue(
            time_value, 2), run_time=0.5)
        self.wait(2)
        for i, nodes in enumerate(nodes):
            self.play(Uncreate(nodes), run_time=0.5)
        self.play(Uncreate(wave1_graph), Unwrite(wavelabel),
                  Unwrite(omega_label), Unwrite(omega2_value), Unwrite(time_label), Unwrite(time_value), Unwrite(caption), Uncreate(ax_1), run_time=2)
        self.wait()


class spherical_coords(ThreeDScene):
    def construct(self):
        # Create axes
        axes = ThreeDAxes()
        # axis labels
        axes_labels = axes.get_axis_labels()
        # Create a vector pointing from the origin to (2,2,1).
        vec = Arrow3D(start=np.array([0, 0, 0]), end=np.array(
            [2, 2, 1]), resolution=8, color=RED)
        vec_label = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 2 \\ 1 \end{bmatrix}").next_to(
            vec, direction=RIGHT, buff=0.1)
        vec_label.rotate(PI/2, axis=np.array([1, 0, 0])).set_color(RED)
        theta_arc = ArcBetweenPoints(start=np.array(
            [0.5, 0.5, 0.25]), end=np.array([0, 0, 0.75]), color=BLUE)
        theta = MathTex(r"\theta", color=BLUE).next_to(
            theta_arc, direction=UP)
        theta.rotate(PI/2, axis=np.array([1, 0, 0]))
        vec_xy = Line3D(start=np.array(
            [0, 0, 0]), end=np.array([2, 2, 0]), color=WHITE)
        vec_xy_label = MathTex(r"|\vec{v}_{xy}| = |\vec{v}| \sin(\theta)", color=WHITE, substrings_to_isolate=["\vec{v}", "\theta"]).next_to(
            vec_xy.get_end(), direction=RIGHT, buff=0.1)
        vec_xy_label.set_color_by_tex("\vec{v}", RED)
        vec_xy_label.set_color_by_tex("\theta", BLUE)
        vec_xy_label.rotate(PI/2, axis=np.array([0, 0, 1]))
        vec_xy_label.rotate(PI/6, axis=np.array([0, 1, 0]))
        dashed = DashedLine(start=vec_xy.get_end(),
                            end=np.array([2, 2, 1]), color=WHITE)
        phi_arc = ArcBetweenPoints(start=np.array(
            [1, 0, 0]), end=np.array([0.707, 0.707, 0]), color=GREEN)
        phi = MathTex(r"\phi", color=GREEN).next_to(phi_arc, direction=RIGHT)
        phi.rotate(PI/2, axis=np.array([1, 0, 0]))
        phi.rotate(PI/2, axis=np.array([0, 0, 1]))
        vec_x = Line3D(start=np.array([0, 0, 0]), end=np.array(
            [2, 0, 0]), color=PINK)
        vec_x_label = MathTex(r"\vec{v}_x = |\vec{v}| \sin(\theta) \cos(\phi)", color=PINK, font_size=30).next_to(
            vec_x, direction=DOWN)
        vec_x_label.rotate(PI/2, axis=np.array([0, 0, 1]))
        vec_x_label.rotate(PI/6, axis=np.array([0, 1, 0]))
        vec_x_label.shift(DOWN, RIGHT)
        vec_y = Line3D(start=np.array([0, 0, 0]), end=np.array(
            [0, 2, 0]), color=PURE_GREEN)
        vec_y_label = MathTex(r"\vec{v}_y = |\vec{v}| \sin(\theta) \sin(\phi)",
                              color=PURE_GREEN, font_size=30).next_to(vec_y, direction=LEFT)
        vec_y_label.rotate(PI, axis=np.array([0, 0, 1]))
        vec_y_label.rotate(-PI/6, axis=np.array([1, 0, 0]))
        vec_z = Line3D(start=np.array([0, 0, 0]), end=np.array(
            [0, 0, 1]), color=PURE_BLUE)
        vec_z_label = MathTex(r"\vec{v}_z = |\vec{v}| \cos(\theta)", color=PURE_BLUE, font_size=30).next_to(
            vec_z, direction=LEFT)
        vec_z_label.rotate(-3*PI/4, axis=np.array([0, 0, 1]))
        vec_z_label.rotate(-PI/2, axis=np.array([1, 1, 0])).shift(LEFT, DOWN)
        vec.save_state()
        vec_xy.save_state()
        dashed.save_state()
        self.move_camera(phi=60 * DEGREES, theta=-60 * DEGREES)
        self.play(Create(axes), Create(axes_labels))
        self.wait(1)
        self.play(Create(vec), Write(vec_label),
                  Create(theta), Create(theta_arc))
        self.wait(2)
        self.play(Uncreate(vec_label), Uncreate(theta))
        self.move_camera(phi=60 * DEGREES, theta=0 * DEGREES, run_time=3)
        self.wait(2)
        self.play(Create(vec_xy), Create(dashed))
        self.wait(1)
        self.play(Create(phi), Create(phi_arc))
        self.wait(1)
        self.play(Indicate(vec_xy), Write(vec_xy_label), run_time=1)
        self.wait(1)
        self.play(Unwrite(vec_xy_label), Uncreate(phi),
                  Uncreate(phi_arc), Uncreate(theta_arc), Uncreate(dashed))
        self.play(ReplacementTransform(vec, vec_xy), run_time=1)
        self.wait(0.5)
        self.play(ReplacementTransform(vec_xy, vec_x),
                  Write(vec_x_label), run_time=1)
        self.wait(1)
        self.remove(vec_x, vec_x_label)
        self.play(Restore(vec), Restore(vec_xy), Restore(dashed))
        self.move_camera(phi=60 * DEGREES, theta=90 * DEGREES, run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(vec, vec_xy),
                  Uncreate(dashed), run_time=1)
        self.wait(0.5)
        self.play(ReplacementTransform(vec_xy, vec_y),
                  Write(vec_y_label), run_time=1)
        self.wait(2)
        self.remove(vec)
        self.remove(vec_y_label)
        self.play(Restore(vec))
        self.move_camera(phi=90 * DEGREES, theta=135 * DEGREES, run_time=2)
        self.wait(1)
        self.play(ReplacementTransform(vec, vec_z),
                  Uncreate(dashed), Write(vec_z_label), run_time=1)
        self.wait(2)
        self.play(Uncreate(vec), Uncreate(vec_z_label), Uncreate(
            axes), Uncreate(axes_labels), Uncreate(vec_z), Uncreate(vec_y), run_time=1)
        self.wait()


class eigenvectors(Scene):
    def construct(self):
        # Create a set of axes.
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_numbers": True}
        )
        # Create the vector [1, 1].
        vec = Vector([2, 1]).set_color(RED)
        vec_label = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}").next_to(
            vec, direction=RIGHT, buff=0.1).set_color(RED)
        # Create the matrix [[2, 0], [0, 2]].
        matrix = Matrix([[2, 0], [0, 2]]).set_color(
            BLUE).shift(DOWN*2, RIGHT*2)
        vec_label_mat = Matrix([[1], [1]]).set_color(
            RED).next_to(matrix, direction=RIGHT, buff=0.1)
        capt = Tex(r"Let's apply the matrix \\[0.9em] $\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ to the vector.", font_size=36).shift(
            UP*2.5, LEFT*3.5)
        capt2 = Tex(r"To get the first entry of the resulting vector,\\ we multiply the first row of the matrix by \\ the vector coordinate-wise and add.",
                    font_size=30).shift(UP*2.5, LEFT*3.5)
        capt3 = Tex(r"To get the second entry of the resulting vector,\\ we multiply the second row of the matrix by \\ the vector coordinate-wise and add.",
                    font_size=30).shift(UP*2.5, LEFT*3.5)
        capt4 = Tex(r"What about another matrix?",
                    font_size=36).shift(UP*2.5, LEFT*3.5)
        capt5 = Tex(r"This time...", font_size=36).shift(UP*2.5, LEFT*3.5)
        capt6 = Tex(r"The vector is not scaled, but rotated. Hence,\\ this vector is not an eigenvector of the green matrix.",
                    font_size=28).shift(UP*2.5, LEFT*3.5)
        # Create the vector [2, 2].
        vec2 = Vector([4, 2]).set_color(RED)
        vec_label2 = MathTex(r"\begin{bmatrix} 2 \\ 2 \end{bmatrix}").next_to(
            vec2, direction=RIGHT, buff=0.1).set_color(RED)
        equals = MathTex(r"=").next_to(
            vec_label_mat, direction=RIGHT, buff=0.2)
        result = Matrix([[2], [2]]).set_color(RED).next_to(
            equals, direction=RIGHT, buff=0.1)
        result_capt = Tex(r"This transformation scales the vector by a \\ factor of 2. Hence, the vector is an eigenvector \\ with eigenvalue 2.",
                          font_size=30).shift(UP*2.5, LEFT*3.5)
        # Create the matrix [[0,-1],[1,0]].
        matrix2 = Matrix([[0, -1], [1, 0]]
                         ).set_color(GREEN).shift(DOWN*2, RIGHT*2)
        # Create the vector [-1, 1].
        vec3 = Vector([-2, 1]).set_color(GREEN)
        vec_label3 = MathTex(r"\begin{bmatrix} -1 \\ 1 \end{bmatrix}").next_to(
            vec3, direction=LEFT, buff=0.1).set_color(GREEN)
        result2 = Matrix([[-1], [1]]).set_color(GREEN).next_to(
            equals, direction=RIGHT, buff=0.1)
        vec.save_state()
        self.play(Create(axes), Create(matrix), Create(
            vec), Write(vec_label), run_time=2)
        self.wait(1)
        self.play(Transform(vec_label, vec_label_mat), Write(capt), run_time=2)
        self.wait(4)
        self.play(ReplacementTransform(capt, capt2), run_time=1)
        self.wait(3)
        self.play(ReplacementTransform(capt2, capt3), run_time=1)
        self.wait(3)
        self.play(ReplacementTransform(vec_label_mat, result),
                  Write(equals), ReplacementTransform(vec, vec2), Write(vec_label2), run_time=2)
        self.wait(3)
        self.remove(vec2)
        equals.save_state()
        self.play(Restore(vec), Unwrite(vec_label2), Uncreate(matrix), Uncreate(
            result), Unwrite(equals), ReplacementTransform(capt3, result_capt), run_time=2)
        self.wait(2)
        self.play(Create(matrix2), ReplacementTransform(
            result_capt, capt4), run_time=2)
        self.wait(2)
        self.play(ReplacementTransform(capt4, capt5), Restore(equals), Create(
            result2), ReplacementTransform(vec, vec3), Write(vec_label3), run_time=3)
        self.wait(1)
        self.play(ReplacementTransform(capt5, capt6), run_time=1)
        self.wait(3)
        self.play(Uncreate(capt6), FadeOut(vec_label3), FadeOut(vec3), Uncreate(matrix2), Uncreate(
            vec_label), Uncreate(equals), Uncreate(result2), FadeOut(axes), run_time=2)
        self.wait()
