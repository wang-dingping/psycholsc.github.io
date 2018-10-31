DATAS SEGMENT
    buf1 db 'abcdefg'
    len1 equ $-buf1
    buf2 db 'bbcdefg'
    len2 equ $-buf2
DATAS ENDS

STACKS SEGMENT
    
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    mov al,len1
    mov bl,len2
    sub al,bl
    JNZ exit_w
    mov cx,len1
    CLD
    lea SI,buf1
    lea DI,buf2
compare:
	cmpsb
	JNZ exit_w
	loop compare
	
exit_r:
	mov al,0
	mov AH,4CH
	INT 21H
    
exit_w: 
	mov al,1
    MOV AH,4CH
    INT 21H

CODES ENDS
    END START

