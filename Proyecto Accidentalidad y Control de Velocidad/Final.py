import Tkinter as tk

from geopy.geocoders import Nominatim
from Fuzzy2 import *
from SE import *
import Backpropagation as nn

#geolocator = Nominatim()
#ubicacion=geolocator.reverse("4.8100127,-75.6907224")
#print(ubicacion.address)

global g_lluvia
global g_carretera

ubicacion="Avenida Circunvalar"



def ventana():
    raiz=tk.Tk()
    raiz.title("Control de Velocidad")
    #raiz.resizable(0,0) #no redimensionable la ventana
    raiz.geometry("470x510") # Tamano de la ventana predetermindado
    
    miFrame = tk.Frame(raiz, width=500, height=400)
    miFrame.pack()
    
    miLabel = tk.Label(miFrame, text="Bienvenido", fg = "blue", font = "Verdana 14 bold")
    miLabel.grid(row=0, column=1, padx=10, pady=10)
    
    lb_lluvia = tk.Label(miFrame, text='Nivel de lluvia', font = "Verdana 11 bold")
    lb_lluvia.grid(row=1, column=0, padx=10, pady=10)
    
    lb_hora = tk.Label(miFrame, text='Hora (Militar)', font = "Verdana 11 bold")
    lb_hora.grid(row=2, column=0, padx=10, pady=10)
    
    lb_dia = tk.Label(miFrame, text='Dia', font = "Verdana 11 bold")
    lb_dia.grid(row=3, column=0, padx=10, pady=10)
    
    lb_carretera = tk.Label(miFrame, text='Carretera', font = "Verdana 11 bold")
    lb_carretera.grid(row=4, column=0, padx=10, pady=10)
    
    lb_mes = tk.Label(miFrame, text='Mes', font = "Verdana 12 bold")
    lb_mes.grid(row=5, column=0, padx=10, pady=10)
    
    lb_latitud = tk.Label(miFrame, text='Latitud', font = "Verdana 12 bold")
    lb_latitud.grid(row=6, column=0, padx=10, pady=10)
    
    lb_longitud = tk.Label(miFrame, text='Longitud', font = "Verdana 12 bold")
    lb_longitud.grid(row=7, column=0, padx=10, pady=10)
    
    lb_avenida = tk.Label(miFrame, text='Avenida', font = "Verdana 12 bold")
    lb_avenida.grid(row=8, column=0, padx=10, pady=10)
    
    lb_result = tk.Label(miFrame, text='---', fg = "red", font = "Verdana 12 bold")
    lb_result.grid(row=10, column=0, padx=10, pady=10)
    
    #miLabel.place(x=150, y =200) #para ubicarlo segun preferencia
    txt_lluvia = tk.Entry(miFrame)
    txt_lluvia.grid(row=1, column=1, padx=10, pady=10)
    txt_lluvia.focus_set()
    
    txt_hora = tk.Entry(miFrame)
    txt_hora.grid(row=2, column=1, padx=10, pady=10)
    txt_hora.focus_set()
    
    txt_dia = tk.Entry(miFrame)
    txt_dia.grid(row=3, column=1, padx=10, pady=10)
    
    txt_dia = tk.Entry(miFrame)
    txt_dia.grid(row=3, column=1, padx=10, pady=10)
    
    txt_carretera = tk.Entry(miFrame)
    txt_carretera.grid(row=4, column=1, padx=10, pady=10)
    txt_carretera.focus_set()
    
    txt_mes = tk.Entry(miFrame)
    txt_mes.grid(row=5, column=1, padx=10, pady=10)
    
    txt_latitud = tk.Entry(miFrame)
    txt_latitud.grid(row=6, column=1, padx=10, pady=10)
    
    txt_longitud = tk.Entry(miFrame)
    txt_longitud.grid(row=7, column=1, padx=10, pady=10)
    
    txt_avenida = tk.Entry(miFrame)
    txt_avenida.grid(row=8, column=1, padx=10, pady=10)
    txt_avenida.focus_set()
    
    
    def boton():
        ll = int(txt_lluvia.get())
        carre = int(txt_carretera.get())
        hour = int(txt_hora.get())
        ubicacion = txt_avenida.get()
        #print(ll)
        
        #Logica Difusa
        riesgo = int(calcular_Riesgo(ll,carre,hour))
        
        d=[]
        if ubicacion == "Avenida del Rio":
            d=[0,0,0,0]
        elif ubicacion == "Avenida 30 Agosto":
        	d=[0,0,0,1]
        elif ubicacion == "Avenida Circunvalar":
        	d=[0,0,1,0]
        elif ubicacion == "Avenida de las Americas":
        	d=[0,0,1,1]
        elif ubicacion == "Avenida Ferrocarril":
        	d=[0,1,0,0]
        elif ubicacion == "Avenida Belalcazar":
        	d=[0,1,0,1]
        elif ubicacion == "Variante La Romelia-El Pollo":
        	d=[0,1,1,0]	
        elif ubicacion == "Avenida Juan B. Gutierrez":
        	d=[0,1,1,1]	
        elif ubicacion == "Carrera 8":
        	d=[1,0,0,0]	
        elif ubicacion == "Carrera 7":
        	d=[1,0,0,1]	
        else:
        	d=[0,0,0,0]	
            
        #Red Neuronal
        binRiesgo=nn.Binario(riesgo)
        entrada=[[[binRiesgo[0],binRiesgo[1],binRiesgo[2],binRiesgo[3],binRiesgo[4],
                   binRiesgo[5],binRiesgo[6],d[0],d[1],d[2],d[3]],[0,0]]]    
        
        accidentalidad=0
        output =nn.red.Resultado(entrada)
        if output[0]==0 and output[1]==0:
    	    accidentalidad=0
        elif output[0]==0 and output[1]==1:
    	    accidentalidad=1
        elif output[0]==1 and output[1]==0:
    	    accidentalidad=2
        elif output[0]==1 and output[1]==1:
    	    accidentalidad=3     
        
        #Sistema Experto
        V(accidentalidad,Velocidad_Sugerida)
        print ("Velocidad Sugerida:",Velocidad_Sugerida.v(),"km/h")
        
        lb_result.config(text="Velocidad Sugerida: " + Velocidad_Sugerida.v() + " km/h")
        
    btn_calcular = tk.Button(miFrame, text="Enviar datos", command=boton)
    btn_calcular.grid(row=9, column=1, padx=10, pady=10)
    

    raiz.mainloop()


def main():
    ventana()

if __name__ == '__main__':
    main()
