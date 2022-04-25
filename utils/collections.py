# from utils.imports import asynckivy


def search_list(self):
    async def search_list():

        search = self.help.get_screen('collections').ids.search.text
        all_docs = db.child("Hoya").get()
        for i in all_docs.each():
            name = i.key()
            if search in name:
                await asynckivy.sleep(0)
                self.help.get_screen('collections').ids.box.add_widget(
                    OneLineIcon(text=f'{name}',
                                on_release=lambda y: self.spin_dialog(),
                                on_press=lambda x, value_for_pass=name: self.passValue_thread(
                                    value_for_pass),

                                ))
    asynckivy.start(search_list())


def search_callback(self, *args):
    '''A method that updates the state of your application
    while the spinner remains on the screen.'''

    def refresh_callback(interval):
        self.help.get_screen('collections').ids.box.clear_widgets()
        self.search_list()
        self.help.get_screen('collections').ids.refresh_layout.refresh_done()
        self.tick = 0

    Clock.schedule_once(refresh_callback, 1)
