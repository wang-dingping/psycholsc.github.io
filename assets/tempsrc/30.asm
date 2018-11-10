DATAS SEGMENT
    A DW 1,2,3,4,5,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33
	B DW 1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32  
	CC DW 20 dup(0)
DATAS ENDS

STACKS SEGMENT
    
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    lea SI,A
    lea DI,B
    mov DX,0
    mov CX,30
l:
	mov AX,[SI]
	mov BX,[DI]
	add DI,2
	cmp AX,BX
	JZ zero
con:
	loop l
	
	
	push SI
	push DI
	lea DI,CC
	add DI,DX
	cld
	movsw
	pop DI
	pop SI
	add SI,2
	add DX,2
	lea DI,B
	mov CX,30
	cmp SI,40
	JL l
	
	mov AH,4CH
	int 21H
zero:
	lea DI,B
	mov CX,30
	add SI,2
	jmp l
CODES ENDS
    END START

