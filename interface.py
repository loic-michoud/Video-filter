from tkinter import *

class Interface(Frame):




    def __init__(self, fenetre, **kwargs):

        self.img = PhotoImage(file='data/lena.gif')
        Frame.__init__(self, fenetre, width=1024, height=768, **kwargs )
        self.pack(fill=BOTH)
        self.backgroung='Green'
        # Création de nos widgets

        self.TextLabel=StringVar()
        self.TextLabel.set("Veuillez séléctionner vos filtres: \n\n"
                                        +"A: Ajouter/Enlever la localisation\n"
                                        +"Z: Filtre Noir&Blanc\n"
                                        +"E: Filtre Rayons X\n"
                                        +"R: Ajouter\Enlever des lunettes \n"
                                        +"T: Ajouter\Enlever des cornes \n"
                                        +"S: Ajouter\Enlever un fond cartoonesque \n"
                                        +"F: Flou Gaussien\n"
                                        +"D: Love is everywhere\n"
                                        +"G: Vitesse Lumière\n"
                                        +"B: Ajouter/Enlever bouche\n"
                                        +"M: Monsieur Macron\n"
                                        +"ESPACE: Faire une photo souvenir :)\n\n"
                                        +"Appuyer sur Q pour quitter l'application")

        self.label=LabelFrame(self, text='Projet Majeure Image réalisé par Mathieu Nicolas & Loïc Michoud',padx=0, pady=0)
        self.label.pack(fill="both", expand="yes")

        self.message = Label(self.label, image=self.img
                                        ,compound = "center",textvariable=self.TextLabel,  font=("Times New Roman", 16),
               justify=LEFT,fg='#99FF99', padx=0, pady=0)

        self.message.pack()
        self.b1 = Button(self, text="START", relief=RAISED,
                                     command=self.cliquer, background='Green')
        self.b1.pack(fill=NONE)

    def cliquer(self):
        print("HAVE FUN ;) ")
        self.quit()


