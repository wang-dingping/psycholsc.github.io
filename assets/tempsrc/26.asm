DATAS SEGMENT
    string db 'data,name,time,file,code,path,user,exit,quit,text$'
    len equ $-string
    buf db 5
    real db ?
    buf2 db 4 dup(0)
    endf db 0DH,0AH,'$'
    disk db 'disk'
DATAS ENDS

STACKS SEGMENT

STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    CLD
    mov ES,AX	; essential
    lea DX,buf
    mov AH,0AH
    int 21H
    mov SI,0
    mov DX,0
se:
	mov CX,5
	mov SI,DX    
    lea DI,buf2
    repz cmpsb	;repz cx ==0 or zf == 0 jmp
	cmp CX,0
    JNZ cn0
    
    lea DX,endf
    mov AH,09H
    int 21H
    lea DX,buf2
    mov AH,09H
    int 21H
    mov CX,4
    
    lea DX,disk
    mov DI,DX
    xchg DI,SI
    sub DI,5
    rep movsb
	
	lea DX,string
	mov AH,09H
	int 21H
  
theend:
    MOV AH,4CH
    INT 21H
    
cn0:
	mov CX,DX
	sub dx,46
	JZ theend
	mov DX,CX
	add DX,1
	JMP se

CODES ENDS
    END START



