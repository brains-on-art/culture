import sdl2
from boa_gfx.gl_core import GLWindow
from boa_gfx.core import EventManager
from boa_gfx.gl_mesh import TriangleStrip, TexturedTriangleStrip
from boa_gfx.gl_ui import ProgressBar
from boa_gfx.transformer import Dynamo, linear, sinusoidal
import boa_gfx.gl_shader
import boa_gfx.gl_texture

from creature import Culture
import culture


class Program(object):
    def __init__(self):
        self.dynamo = Dynamo()

        self.event_manager = EventManager()
        self.windows = []

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError('SDL_Init failed: {}'.format(sdl2.SDL_GetError()))

    def main_loop(self):
        running = True

        def toggle_running(event):
            print('Quit event detected')
            nonlocal running
            running = not running

        #print(sdl2.SDL_GL_GetSwapInterval())

        self.event_manager.add_quit_callback(toggle_running)
        self.event_manager.add_keydown_callback(sdl2.SDLK_q, toggle_running)
        boa_gfx.gl_shader.shader_manager.shader_paths.append('./boa_gfx/shaders')
        boa_gfx.gl_shader.shader_manager.shader_paths.append('./shaders')
        boa_gfx.gl_texture.texture_manager.texture_paths.append('./textures')

        self.windows[0].camera.position = (0.0, 0.0, 13.0)
        self.windows[0].fullscreen = True

        background = TexturedTriangleStrip(texture_name='background.png')
        background.scale = 13

        #c_old = Culture(self.event_manager)
        #for i in range(100):
        #    c_old.add_creature()

        c = culture.Culture()
        d = culture.Animations()
        #c.position = (0.0, 0.0, 0.)

        #u = HaloCreature()
        #print('u vertex data:', u.vertex_data)
        #u.position += (3.0, 2.0, 0.01)

        #v = WormCreature()
        #v.position += (-5.0, 0.0, 0.01)
        #v.z_rotation = 1.0
        #v.scale = 2.0

        while running:
            self.dynamo.run()
            self.event_manager.process_events()

            #u.update()
            #v# .update()
            #c_old.update()
            c.update(self.dynamo.dt)
            d.update(self.dynamo.dt)

            for w in self.windows:
                w.render()
            sdl2.SDL_Delay(1)

        for w in self.windows:
            w.close()
        sdl2.SDL_Quit()


if __name__ == '__main__':
    prog = Program()
    prog.windows.append(GLWindow(prog))

    prog.main_loop()
