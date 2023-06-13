import pygame as pg
import sys
import random

def bola_animaçao():
    global bola_vx,bola_vy,play1_score,play2_score
    # Atualizando as posições
    bola.x += bola_vx     
    bola.y += bola_vy  
    # Condições para as paredes :
    if pg.Rect.colliderect(bola,parede_down) or pg.Rect.colliderect(bola,parede_top):
        bola_vy *=-1  
    # Condições para os playes :
    if pg.Rect.colliderect(bola,play_1):
        if bola_vy//play_speed1 >= 0 :
            bola_vx *=-1 
        else:
            bola_vx *=-1 
            bola_vy *=-1 
    if pg.Rect.colliderect(bola,play_2):
        bola_vx *=-1 
    # se a bola sair :
    if bola.x<0 or bola.x>largura:
        if bola.x>largura :
            play1_score += 1
        if bola.x<0 <= 0 :
            play2_score += 1
        bola_restart()
def play1_animação():
    global play_speed1, altura  
    
    # aplicandi momento na bola
    if play_speed1 > 1 :
        play_speed1 = 1
    if play_speed1 <-1:
        play_speed1 =- 1
    # limitando o movimento do player
    if play_1.top <= 0 :
        play_1.top = 0 
    if play_1.bottom >= altura :
        play_1.bottom = altura    
    if play_2.top <= 0 :
        play_2.top = 0 
    if play_2.bottom >= altura :
        play_2.bottom = altura 

def play2_animation():
    if bola.x <= largura//2 :
        if bola.y <= play_2.y:
            play_2.y-=10
        if bola.y > play_2.y:
            play_2.y+=10

def bola_restart():
    global bola_vx,bola_vy
    bola.center = (largura/2, altura/2 )
    bola_vx = random.choice((-1,1))*5
    bola_vy = random.choice((-1,1))*5


pg.display.init()
relogio = pg.time.Clock()
logo    = pg.image.load("c:/Users/Oibug/Desktop/gubio/Outros/Criando jogos/Python/Pong gamer/logo_mini.jpg")
fundo   = pg.image.load("c:/Users/Oibug/Desktop/gubio/Outros/Criando jogos/Python/Pong gamer/backgroud.jpg")
pg.display.set_icon(logo)
pg.display.set_caption('Pong')


# configurando o tamanho da tela
largura = 1280  
altura  = 720 
tela    = pg.display.set_mode((largura,altura))# width, height

# Criando os objetos :
bola   = pg.Rect(largura/2-24/2, altura/2-25/2 , 15, 15)# Coordenadas da bola : x = 512 ; y = 335
play_1 = pg.Rect(largura-20    , altura/2-50   , 10, 100)# Coordenadas do play1 :  x = 1270 ; y = 310
play_2 = pg.Rect(0      +10    , altura/2-50   , 10, 100)# Coordenadas do play2 : x = 0 ; y = 310

L_parede = 2                                                 # largura da parede
parede_top = pg.Rect(0, 0              , largura, L_parede )
parede_down= pg.Rect(0, altura-L_parede, largura, L_parede )
linha_cent = pg.Rect(largura//2, 0     , 2      , altura   )  #

# Velocidade da bola :

bola_vx = random.choice((-1,1))*5
bola_vy = random.choice((-1,1))*5
play_speed1=0


pg.font.init()
gamer_font = pg.font.Font(None, 40)
play1_score = 0
play2_score = 0
a = True
while a:
    #tela.blit(fundo,(0,0))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()        
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_DOWN:
                play_1.y += 75 
                play_speed1+=2
            if event.key == pg.K_UP:
                play_1.y -= 75 
                play_speed1-=2
                
    bola_animaçao()
    play1_animação()
    play2_animation()
    # Desenhando os objetos:  
    tela.fill("black")      
    pg.draw.rect(   tela, pg.Color('white'),linha_cent)
    pg.draw.ellipse(tela, pg.Color('gray') , bola    )
    pg.draw.rect(   tela, pg.Color('blue') , play_1  )
    pg.draw.rect(   tela, pg.Color('red')  , play_2  )
    pg.draw.rect(   tela, pg.Color("white"),parede_top)
    pg.draw.rect(   tela, pg.Color("white"),parede_down)
    
    texto = gamer_font.render(f"{play1_score}",False,pg.Color('white'))   
    texto2 = gamer_font.render(f'{play2_score}',False,pg.Color('white'))   
    
    tela.blit(texto,(largura//2 + 40,altura//2) )
    tela.blit(texto2,(largura//2 - 50,altura//2) )
    # Atualização da tela 
    pg.display.flip()
    relogio.tick(60)
    
#pg.quit()