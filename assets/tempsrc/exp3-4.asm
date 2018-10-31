DATAS SEGMENT
    buf db 'Can you find # in the string?$'
    count equ $-buf
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    lea DI,buf
    mov AL,'#'
    mov CX,count
    CLD
    repnz scasb
    INC CX
    DEC CX

    JZ notfound
    JNZ found
notfound:
	mov Al,1
	mov AH,4CH
	INT 21H
found:
    mov AL,0
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START

