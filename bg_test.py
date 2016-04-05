import sdl2
from boa_gfx.gl_core import GLWindow
from boa_gfx.core import EventManager
from boa_gfx.gl_mesh import TexturedTriangleStrip
from boa_gfx.transformer import Dynamo
import boa_gfx.gl_shader
import boa_gfx.gl_texture

from boa_gfx.time import TimeKeeper


class Program(object):
    def __init__(self):
        self.dynamo = Dynamo()

        self.event_manager = EventManager()
        self.windows = []
        self.time_keeper = TimeKeeper()
        self.time_keeper.update()

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError('SDL_Init failed: {}'.format(sdl2.SDL_GetError()))

    def main_loop(self):
        running = True

        def toggle_running(event):
            print('Quit event detected')
            nonlocal running
            running = not running


        self.event_manager.add_quit_callback(toggle_running)
        self.event_manager.add_keydown_callback(sdl2.SDLK_q, toggle_running)
        boa_gfx.gl_shader.shader_manager.shader_paths.append('./boa_gfx/shaders')
        boa_gfx.gl_shader.shader_manager.shader_paths.append('./shaders')
        boa_gfx.gl_texture.texture_manager.texture_paths.append('./source_images')

        self.windows[0].camera.position = (-0.5, 0.0, 13.0)
        self.windows[0].fullscreen = True

        background = TexturedTriangleStrip(texture_name='background.png')
        background.position = (0.0, 0.0, 0.0)
        background.scale = 13


        while running:
            self.time_keeper.update()
            self.dynamo.run()
            self.event_manager.process_events()

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
