ЅЫ.
Ў ® 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
Ѓ
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЌћL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8СЄ)
°
$Adam/vae/nyan_decoder/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*5
shared_name&$Adam/vae/nyan_decoder/dense_5/bias/v
Ъ
8Adam/vae/nyan_decoder/dense_5/bias/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_5/bias/v*
_output_shapes	
:Ќ*
dtype0
™
&Adam/vae/nyan_decoder/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АЌ*7
shared_name(&Adam/vae/nyan_decoder/dense_5/kernel/v
£
:Adam/vae/nyan_decoder/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_5/kernel/v* 
_output_shapes
:
АЌ*
dtype0
°
$Adam/vae/nyan_decoder/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:І*5
shared_name&$Adam/vae/nyan_decoder/dense_4/bias/v
Ъ
8Adam/vae/nyan_decoder/dense_4/bias/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_4/bias/v*
_output_shapes	
:І*
dtype0
™
&Adam/vae/nyan_decoder/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АІ*7
shared_name(&Adam/vae/nyan_decoder/dense_4/kernel/v
£
:Adam/vae/nyan_decoder/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_4/kernel/v* 
_output_shapes
:
АІ*
dtype0
°
$Adam/vae/nyan_decoder/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_decoder/dense_3/bias/v
Ъ
8Adam/vae/nyan_decoder/dense_3/bias/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_3/bias/v*
_output_shapes	
:А*
dtype0
©
&Adam/vae/nyan_decoder/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/vae/nyan_decoder/dense_3/kernel/v
Ґ
:Adam/vae/nyan_decoder/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_3/kernel/v*
_output_shapes
:	@А*
dtype0
§
&Adam/vae/nyan_encoder/z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/vae/nyan_encoder/z_log_var/bias/v
Э
:Adam/vae/nyan_encoder/z_log_var/bias/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/z_log_var/bias/v*
_output_shapes
:@*
dtype0
≠
(Adam/vae/nyan_encoder/z_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*9
shared_name*(Adam/vae/nyan_encoder/z_log_var/kernel/v
¶
<Adam/vae/nyan_encoder/z_log_var/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/vae/nyan_encoder/z_log_var/kernel/v*
_output_shapes
:	А@*
dtype0
Ю
#Adam/vae/nyan_encoder/z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/vae/nyan_encoder/z_mean/bias/v
Ч
7Adam/vae/nyan_encoder/z_mean/bias/v/Read/ReadVariableOpReadVariableOp#Adam/vae/nyan_encoder/z_mean/bias/v*
_output_shapes
:@*
dtype0
І
%Adam/vae/nyan_encoder/z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*6
shared_name'%Adam/vae/nyan_encoder/z_mean/kernel/v
†
9Adam/vae/nyan_encoder/z_mean/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/vae/nyan_encoder/z_mean/kernel/v*
_output_shapes
:	А@*
dtype0
°
$Adam/vae/nyan_encoder/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_encoder/dense_2/bias/v
Ъ
8Adam/vae/nyan_encoder/dense_2/bias/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense_2/bias/v*
_output_shapes	
:А*
dtype0
™
&Adam/vae/nyan_encoder/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*7
shared_name(&Adam/vae/nyan_encoder/dense_2/kernel/v
£
:Adam/vae/nyan_encoder/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/dense_2/kernel/v* 
_output_shapes
:
АА*
dtype0
°
$Adam/vae/nyan_encoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_encoder/dense_1/bias/v
Ъ
8Adam/vae/nyan_encoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense_1/bias/v*
_output_shapes	
:А*
dtype0
©
&Adam/vae/nyan_encoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/vae/nyan_encoder/dense_1/kernel/v
Ґ
:Adam/vae/nyan_encoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/dense_1/kernel/v*
_output_shapes
:	 А*
dtype0
Ј
/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v
∞
CAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v*
_output_shapes	
:А*
dtype0
њ
1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*B
shared_name31Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v
Є
EAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v*
_output_shapes
:	А*
dtype0
¶
'Adam/vae/nyan_encoder/ecc_conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/vae/nyan_encoder/ecc_conv_2/bias/v
Я
;Adam/vae/nyan_encoder/ecc_conv_2/bias/v/Read/ReadVariableOpReadVariableOp'Adam/vae/nyan_encoder/ecc_conv_2/bias/v*
_output_shapes
: *
dtype0
Є
.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/v
±
BAdam/vae/nyan_encoder/ecc_conv_2/root_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/v*
_output_shapes

:  *
dtype0
Ј
/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v
∞
CAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v*
_output_shapes	
:А*
dtype0
њ
1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*B
shared_name31Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v
Є
EAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v*
_output_shapes
:	А*
dtype0
¶
'Adam/vae/nyan_encoder/ecc_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/vae/nyan_encoder/ecc_conv_1/bias/v
Я
;Adam/vae/nyan_encoder/ecc_conv_1/bias/v/Read/ReadVariableOpReadVariableOp'Adam/vae/nyan_encoder/ecc_conv_1/bias/v*
_output_shapes
: *
dtype0
Є
.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/v
±
BAdam/vae/nyan_encoder/ecc_conv_1/root_kernel/v/Read/ReadVariableOpReadVariableOp.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/v*
_output_shapes

:  *
dtype0
≥
-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v
ђ
AAdam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v/Read/ReadVariableOpReadVariableOp-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v*
_output_shapes	
:А*
dtype0
ї
/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v
і
CAdam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v*
_output_shapes
:	А*
dtype0
Ґ
%Adam/vae/nyan_encoder/ecc_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/vae/nyan_encoder/ecc_conv/bias/v
Ы
9Adam/vae/nyan_encoder/ecc_conv/bias/v/Read/ReadVariableOpReadVariableOp%Adam/vae/nyan_encoder/ecc_conv/bias/v*
_output_shapes
: *
dtype0
і
,Adam/vae/nyan_encoder/ecc_conv/root_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/vae/nyan_encoder/ecc_conv/root_kernel/v
≠
@Adam/vae/nyan_encoder/ecc_conv/root_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/vae/nyan_encoder/ecc_conv/root_kernel/v*
_output_shapes

: *
dtype0
Ь
"Adam/vae/nyan_encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/vae/nyan_encoder/dense/bias/v
Х
6Adam/vae/nyan_encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp"Adam/vae/nyan_encoder/dense/bias/v*
_output_shapes
:*
dtype0
§
$Adam/vae/nyan_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/vae/nyan_encoder/dense/kernel/v
Э
8Adam/vae/nyan_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense/kernel/v*
_output_shapes

:*
dtype0
°
$Adam/vae/nyan_decoder/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*5
shared_name&$Adam/vae/nyan_decoder/dense_5/bias/m
Ъ
8Adam/vae/nyan_decoder/dense_5/bias/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_5/bias/m*
_output_shapes	
:Ќ*
dtype0
™
&Adam/vae/nyan_decoder/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АЌ*7
shared_name(&Adam/vae/nyan_decoder/dense_5/kernel/m
£
:Adam/vae/nyan_decoder/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_5/kernel/m* 
_output_shapes
:
АЌ*
dtype0
°
$Adam/vae/nyan_decoder/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:І*5
shared_name&$Adam/vae/nyan_decoder/dense_4/bias/m
Ъ
8Adam/vae/nyan_decoder/dense_4/bias/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_4/bias/m*
_output_shapes	
:І*
dtype0
™
&Adam/vae/nyan_decoder/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АІ*7
shared_name(&Adam/vae/nyan_decoder/dense_4/kernel/m
£
:Adam/vae/nyan_decoder/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_4/kernel/m* 
_output_shapes
:
АІ*
dtype0
°
$Adam/vae/nyan_decoder/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_decoder/dense_3/bias/m
Ъ
8Adam/vae/nyan_decoder/dense_3/bias/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_decoder/dense_3/bias/m*
_output_shapes	
:А*
dtype0
©
&Adam/vae/nyan_decoder/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/vae/nyan_decoder/dense_3/kernel/m
Ґ
:Adam/vae/nyan_decoder/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_decoder/dense_3/kernel/m*
_output_shapes
:	@А*
dtype0
§
&Adam/vae/nyan_encoder/z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&Adam/vae/nyan_encoder/z_log_var/bias/m
Э
:Adam/vae/nyan_encoder/z_log_var/bias/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/z_log_var/bias/m*
_output_shapes
:@*
dtype0
≠
(Adam/vae/nyan_encoder/z_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*9
shared_name*(Adam/vae/nyan_encoder/z_log_var/kernel/m
¶
<Adam/vae/nyan_encoder/z_log_var/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/vae/nyan_encoder/z_log_var/kernel/m*
_output_shapes
:	А@*
dtype0
Ю
#Adam/vae/nyan_encoder/z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/vae/nyan_encoder/z_mean/bias/m
Ч
7Adam/vae/nyan_encoder/z_mean/bias/m/Read/ReadVariableOpReadVariableOp#Adam/vae/nyan_encoder/z_mean/bias/m*
_output_shapes
:@*
dtype0
І
%Adam/vae/nyan_encoder/z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*6
shared_name'%Adam/vae/nyan_encoder/z_mean/kernel/m
†
9Adam/vae/nyan_encoder/z_mean/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/vae/nyan_encoder/z_mean/kernel/m*
_output_shapes
:	А@*
dtype0
°
$Adam/vae/nyan_encoder/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_encoder/dense_2/bias/m
Ъ
8Adam/vae/nyan_encoder/dense_2/bias/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense_2/bias/m*
_output_shapes	
:А*
dtype0
™
&Adam/vae/nyan_encoder/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*7
shared_name(&Adam/vae/nyan_encoder/dense_2/kernel/m
£
:Adam/vae/nyan_encoder/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/dense_2/kernel/m* 
_output_shapes
:
АА*
dtype0
°
$Adam/vae/nyan_encoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adam/vae/nyan_encoder/dense_1/bias/m
Ъ
8Adam/vae/nyan_encoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense_1/bias/m*
_output_shapes	
:А*
dtype0
©
&Adam/vae/nyan_encoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/vae/nyan_encoder/dense_1/kernel/m
Ґ
:Adam/vae/nyan_encoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/vae/nyan_encoder/dense_1/kernel/m*
_output_shapes
:	 А*
dtype0
Ј
/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m
∞
CAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m*
_output_shapes	
:А*
dtype0
њ
1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*B
shared_name31Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m
Є
EAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m*
_output_shapes
:	А*
dtype0
¶
'Adam/vae/nyan_encoder/ecc_conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/vae/nyan_encoder/ecc_conv_2/bias/m
Я
;Adam/vae/nyan_encoder/ecc_conv_2/bias/m/Read/ReadVariableOpReadVariableOp'Adam/vae/nyan_encoder/ecc_conv_2/bias/m*
_output_shapes
: *
dtype0
Є
.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/m
±
BAdam/vae/nyan_encoder/ecc_conv_2/root_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/m*
_output_shapes

:  *
dtype0
Ј
/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m
∞
CAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m*
_output_shapes	
:А*
dtype0
њ
1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*B
shared_name31Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m
Є
EAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m*
_output_shapes
:	А*
dtype0
¶
'Adam/vae/nyan_encoder/ecc_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/vae/nyan_encoder/ecc_conv_1/bias/m
Я
;Adam/vae/nyan_encoder/ecc_conv_1/bias/m/Read/ReadVariableOpReadVariableOp'Adam/vae/nyan_encoder/ecc_conv_1/bias/m*
_output_shapes
: *
dtype0
Є
.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/m
±
BAdam/vae/nyan_encoder/ecc_conv_1/root_kernel/m/Read/ReadVariableOpReadVariableOp.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/m*
_output_shapes

:  *
dtype0
≥
-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m
ђ
AAdam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m/Read/ReadVariableOpReadVariableOp-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m*
_output_shapes	
:А*
dtype0
ї
/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*@
shared_name1/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m
і
CAdam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m*
_output_shapes
:	А*
dtype0
Ґ
%Adam/vae/nyan_encoder/ecc_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/vae/nyan_encoder/ecc_conv/bias/m
Ы
9Adam/vae/nyan_encoder/ecc_conv/bias/m/Read/ReadVariableOpReadVariableOp%Adam/vae/nyan_encoder/ecc_conv/bias/m*
_output_shapes
: *
dtype0
і
,Adam/vae/nyan_encoder/ecc_conv/root_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/vae/nyan_encoder/ecc_conv/root_kernel/m
≠
@Adam/vae/nyan_encoder/ecc_conv/root_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/vae/nyan_encoder/ecc_conv/root_kernel/m*
_output_shapes

: *
dtype0
Ь
"Adam/vae/nyan_encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/vae/nyan_encoder/dense/bias/m
Х
6Adam/vae/nyan_encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp"Adam/vae/nyan_encoder/dense/bias/m*
_output_shapes
:*
dtype0
§
$Adam/vae/nyan_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/vae/nyan_encoder/dense/kernel/m
Э
8Adam/vae/nyan_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/vae/nyan_encoder/dense/kernel/m*
_output_shapes

:*
dtype0
Т
vae/nyan_encoder/sampling/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!vae/nyan_encoder/sampling/count
Л
3vae/nyan_encoder/sampling/count/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/sampling/count*
_output_shapes
: *
dtype0
Т
vae/nyan_encoder/sampling/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!vae/nyan_encoder/sampling/total
Л
3vae/nyan_encoder/sampling/total/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/sampling/total*
_output_shapes
: *
dtype0
Ц
!vae/nyan_encoder/sampling/count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!vae/nyan_encoder/sampling/count_1
П
5vae/nyan_encoder/sampling/count_1/Read/ReadVariableOpReadVariableOp!vae/nyan_encoder/sampling/count_1*
_output_shapes
: *
dtype0
Ц
!vae/nyan_encoder/sampling/total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!vae/nyan_encoder/sampling/total_1
П
5vae/nyan_encoder/sampling/total_1/Read/ReadVariableOpReadVariableOp!vae/nyan_encoder/sampling/total_1*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
global_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameglobal_step
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0
У
vae/nyan_decoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*.
shared_namevae/nyan_decoder/dense_5/bias
М
1vae/nyan_decoder/dense_5/bias/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_5/bias*
_output_shapes	
:Ќ*
dtype0
Ь
vae/nyan_decoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АЌ*0
shared_name!vae/nyan_decoder/dense_5/kernel
Х
3vae/nyan_decoder/dense_5/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_5/kernel* 
_output_shapes
:
АЌ*
dtype0
У
vae/nyan_decoder/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:І*.
shared_namevae/nyan_decoder/dense_4/bias
М
1vae/nyan_decoder/dense_4/bias/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_4/bias*
_output_shapes	
:І*
dtype0
Ь
vae/nyan_decoder/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АІ*0
shared_name!vae/nyan_decoder/dense_4/kernel
Х
3vae/nyan_decoder/dense_4/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_4/kernel* 
_output_shapes
:
АІ*
dtype0
У
vae/nyan_decoder/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namevae/nyan_decoder/dense_3/bias
М
1vae/nyan_decoder/dense_3/bias/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_3/bias*
_output_shapes	
:А*
dtype0
Ы
vae/nyan_decoder/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*0
shared_name!vae/nyan_decoder/dense_3/kernel
Ф
3vae/nyan_decoder/dense_3/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_decoder/dense_3/kernel*
_output_shapes
:	@А*
dtype0
Ц
vae/nyan_encoder/z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!vae/nyan_encoder/z_log_var/bias
П
3vae/nyan_encoder/z_log_var/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/z_log_var/bias*
_output_shapes
:@*
dtype0
Я
!vae/nyan_encoder/z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*2
shared_name#!vae/nyan_encoder/z_log_var/kernel
Ш
5vae/nyan_encoder/z_log_var/kernel/Read/ReadVariableOpReadVariableOp!vae/nyan_encoder/z_log_var/kernel*
_output_shapes
:	А@*
dtype0
Р
vae/nyan_encoder/z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namevae/nyan_encoder/z_mean/bias
Й
0vae/nyan_encoder/z_mean/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/z_mean/bias*
_output_shapes
:@*
dtype0
Щ
vae/nyan_encoder/z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*/
shared_name vae/nyan_encoder/z_mean/kernel
Т
2vae/nyan_encoder/z_mean/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/z_mean/kernel*
_output_shapes
:	А@*
dtype0
У
vae/nyan_encoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namevae/nyan_encoder/dense_2/bias
М
1vae/nyan_encoder/dense_2/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense_2/bias*
_output_shapes	
:А*
dtype0
Ь
vae/nyan_encoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*0
shared_name!vae/nyan_encoder/dense_2/kernel
Х
3vae/nyan_encoder/dense_2/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense_2/kernel* 
_output_shapes
:
АА*
dtype0
У
vae/nyan_encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namevae/nyan_encoder/dense_1/bias
М
1vae/nyan_encoder/dense_1/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense_1/bias*
_output_shapes	
:А*
dtype0
Ы
vae/nyan_encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*0
shared_name!vae/nyan_encoder/dense_1/kernel
Ф
3vae/nyan_encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense_1/kernel*
_output_shapes
:	 А*
dtype0
©
(vae/nyan_encoder/ecc_conv_2/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(vae/nyan_encoder/ecc_conv_2/FGN_out/bias
Ґ
<vae/nyan_encoder/ecc_conv_2/FGN_out/bias/Read/ReadVariableOpReadVariableOp(vae/nyan_encoder/ecc_conv_2/FGN_out/bias*
_output_shapes	
:А*
dtype0
±
*vae/nyan_encoder/ecc_conv_2/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*;
shared_name,*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel
™
>vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/Read/ReadVariableOpReadVariableOp*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel*
_output_shapes
:	А*
dtype0
Ш
 vae/nyan_encoder/ecc_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" vae/nyan_encoder/ecc_conv_2/bias
С
4vae/nyan_encoder/ecc_conv_2/bias/Read/ReadVariableOpReadVariableOp vae/nyan_encoder/ecc_conv_2/bias*
_output_shapes
: *
dtype0
™
'vae/nyan_encoder/ecc_conv_2/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *8
shared_name)'vae/nyan_encoder/ecc_conv_2/root_kernel
£
;vae/nyan_encoder/ecc_conv_2/root_kernel/Read/ReadVariableOpReadVariableOp'vae/nyan_encoder/ecc_conv_2/root_kernel*
_output_shapes

:  *
dtype0
©
(vae/nyan_encoder/ecc_conv_1/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(vae/nyan_encoder/ecc_conv_1/FGN_out/bias
Ґ
<vae/nyan_encoder/ecc_conv_1/FGN_out/bias/Read/ReadVariableOpReadVariableOp(vae/nyan_encoder/ecc_conv_1/FGN_out/bias*
_output_shapes	
:А*
dtype0
±
*vae/nyan_encoder/ecc_conv_1/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*;
shared_name,*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel
™
>vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/Read/ReadVariableOpReadVariableOp*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel*
_output_shapes
:	А*
dtype0
Ш
 vae/nyan_encoder/ecc_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" vae/nyan_encoder/ecc_conv_1/bias
С
4vae/nyan_encoder/ecc_conv_1/bias/Read/ReadVariableOpReadVariableOp vae/nyan_encoder/ecc_conv_1/bias*
_output_shapes
: *
dtype0
™
'vae/nyan_encoder/ecc_conv_1/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *8
shared_name)'vae/nyan_encoder/ecc_conv_1/root_kernel
£
;vae/nyan_encoder/ecc_conv_1/root_kernel/Read/ReadVariableOpReadVariableOp'vae/nyan_encoder/ecc_conv_1/root_kernel*
_output_shapes

:  *
dtype0
•
&vae/nyan_encoder/ecc_conv/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&vae/nyan_encoder/ecc_conv/FGN_out/bias
Ю
:vae/nyan_encoder/ecc_conv/FGN_out/bias/Read/ReadVariableOpReadVariableOp&vae/nyan_encoder/ecc_conv/FGN_out/bias*
_output_shapes	
:А*
dtype0
≠
(vae/nyan_encoder/ecc_conv/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(vae/nyan_encoder/ecc_conv/FGN_out/kernel
¶
<vae/nyan_encoder/ecc_conv/FGN_out/kernel/Read/ReadVariableOpReadVariableOp(vae/nyan_encoder/ecc_conv/FGN_out/kernel*
_output_shapes
:	А*
dtype0
Ф
vae/nyan_encoder/ecc_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name vae/nyan_encoder/ecc_conv/bias
Н
2vae/nyan_encoder/ecc_conv/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/ecc_conv/bias*
_output_shapes
: *
dtype0
¶
%vae/nyan_encoder/ecc_conv/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%vae/nyan_encoder/ecc_conv/root_kernel
Я
9vae/nyan_encoder/ecc_conv/root_kernel/Read/ReadVariableOpReadVariableOp%vae/nyan_encoder/ecc_conv/root_kernel*
_output_shapes

: *
dtype0
О
vae/nyan_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namevae/nyan_encoder/dense/bias
З
/vae/nyan_encoder/dense/bias/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense/bias*
_output_shapes
:*
dtype0
Ц
vae/nyan_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namevae/nyan_encoder/dense/kernel
П
1vae/nyan_encoder/dense/kernel/Read/ReadVariableOpReadVariableOpvae/nyan_encoder/dense/kernel*
_output_shapes

:*
dtype0
n
global_step_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameglobal_step_1
g
!global_step_1/Read/ReadVariableOpReadVariableOpglobal_step_1*
_output_shapes
: *
dtype0
В
serving_default_input_1Placeholder*+
_output_shapes
:€€€€€€€€€<*
dtype0* 
shape:€€€€€€€€€<
В
serving_default_input_2Placeholder*+
_output_shapes
:€€€€€€€€€<<*
dtype0	* 
shape:€€€€€€€€€<<
К
serving_default_input_3Placeholder*/
_output_shapes
:€€€€€€€€€<<*
dtype0*$
shape:€€€€€€€€€<<
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3vae/nyan_encoder/dense/kernelvae/nyan_encoder/dense/bias(vae/nyan_encoder/ecc_conv/FGN_out/kernel&vae/nyan_encoder/ecc_conv/FGN_out/bias%vae/nyan_encoder/ecc_conv/root_kernelvae/nyan_encoder/ecc_conv/bias*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel(vae/nyan_encoder/ecc_conv_1/FGN_out/bias'vae/nyan_encoder/ecc_conv_1/root_kernel vae/nyan_encoder/ecc_conv_1/bias*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel(vae/nyan_encoder/ecc_conv_2/FGN_out/bias'vae/nyan_encoder/ecc_conv_2/root_kernel vae/nyan_encoder/ecc_conv_2/biasvae/nyan_encoder/dense_1/kernelvae/nyan_encoder/dense_1/biasvae/nyan_encoder/dense_2/kernelvae/nyan_encoder/dense_2/biasvae/nyan_encoder/z_mean/kernelvae/nyan_encoder/z_mean/bias!vae/nyan_encoder/z_log_var/kernelvae/nyan_encoder/z_log_var/biasglobal_step_1!vae/nyan_encoder/sampling/total_1!vae/nyan_encoder/sampling/count_1vae/nyan_encoder/sampling/totalvae/nyan_encoder/sampling/countvae/nyan_decoder/dense_3/kernelvae/nyan_decoder/dense_3/biasvae/nyan_decoder/dense_4/kernelvae/nyan_decoder/dense_4/biasvae/nyan_decoder/dense_5/kernelvae/nyan_decoder/dense_5/bias*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_7409409

NoOpNoOp
 ф
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Дф
valueщуBху Bну
т
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder


epochs
	optimizer

signatures*
к
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21

22
#23
$24
%25
&26
'27
(28
)29*
Џ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24
&25
'26
(27*
* 
∞
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
* 
ш
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=input_names
>output_names


epochs
?masking
@precondition
A
graphconv1
B
graphconv2
C
graphconv3
	Dpool1

Edense1
Fflatten

Gdense2

Hz_mean
I	z_log_var
Jlatent_z
Knyan_layers*
Џ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

)epochs

Rdense3
Sfingerprint
T
regression
Unyan_layers*
HB
VARIABLE_VALUEglobal_step_1!epochs/.ATTRIBUTES/VARIABLE_VALUE*
б
Viter

Wbeta_1

Xbeta_2
	Ydecaym€mАmБmВmГmДmЕmЖmЗmИmЙmКmЛmМmНmОmПmРmС mТ!mУ"mФ#mХ$mЦ%mЧ&mШ'mЩ(mЪvЫvЬvЭvЮvЯv†v°vҐv£v§v•v¶vІv®v©v™vЂvђv≠ vЃ!vѓ"v∞#v±$v≤%v≥&vі'vµ(vґ*

Zserving_default* 
]W
VARIABLE_VALUEvae/nyan_encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEvae/nyan_encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%vae/nyan_encoder/ecc_conv/root_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_encoder/ecc_conv/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(vae/nyan_encoder/ecc_conv/FGN_out/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&vae/nyan_encoder/ecc_conv/FGN_out/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'vae/nyan_encoder/ecc_conv_1/root_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE vae/nyan_encoder/ecc_conv_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(vae/nyan_encoder/ecc_conv_1/FGN_out/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'vae/nyan_encoder/ecc_conv_2/root_kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE vae/nyan_encoder/ecc_conv_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(vae/nyan_encoder/ecc_conv_2/FGN_out/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_encoder/dense_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_encoder/dense_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_encoder/dense_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_encoder/dense_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEvae/nyan_encoder/z_mean/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEvae/nyan_encoder/z_mean/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!vae/nyan_encoder/z_log_var/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_encoder/z_log_var/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_decoder/dense_3/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_decoder/dense_3/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_decoder/dense_4/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_decoder/dense_4/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvae/nyan_decoder/dense_5/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEvae/nyan_decoder/dense_5/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEglobal_step'variables/29/.ATTRIBUTES/VARIABLE_VALUE*


0
)1*

0
	1*
'
[0
\1
]2
^3
_4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
≤
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21

22*
™
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21*
* 
У
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
6
etrace_0
ftrace_1
gtrace_2
htrace_3* 
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
* 
* 
О
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
ґ
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y
activation

kernel
bias*
к
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkwargs_keys
Б
activation
Вkernel_network_layers
root_kernel
bias*
р
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkwargs_keys
К
activation
Лkernel_network_layers
root_kernel
bias*
р
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkwargs_keys
У
activation
Фkernel_network_layers
root_kernel
bias*
Ф
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses* 
љ
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°
activation

kernel
bias*
Ф
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses* 
љ
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses
Ѓ
activation

kernel
bias*
ђ
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses

kernel
 bias*
ђ
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses

!kernel
"bias*
Ц
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses*
f
Ѕ0
¬1
√2
ƒ3
≈4
∆5
«6
»7
…8
 9
Ћ10
ћ11*
5
#0
$1
%2
&3
'4
(5
)6*
.
#0
$1
%2
&3
'4
(5*
* 
Ш
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

“trace_0
”trace_1* 

‘trace_0
’trace_1* 
љ
÷	variables
„trainable_variables
Ўregularization_losses
ў	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses
№
activation

#kernel
$bias*
ђ
Ё	variables
ёtrainable_variables
яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

%kernel
&bias*
ђ
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses

'kernel
(bias*
$
й0
к1
л2
м3*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
н	variables
о	keras_api

пtotal

рcount*
M
с	variables
т	keras_api

уtotal

фcount
х
_fn_kwargs*
M
ц	variables
ч	keras_api

шtotal

щcount
ъ
_fn_kwargs*
<
ы	variables
ь	keras_api

эtotal

юcount*
<
€	variables
А	keras_api

Бtotal

Вcount*


0*
Z
?0
@1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11*

^0
_1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ц
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

Иtrace_0
Йtrace_1* 

Кtrace_0
Лtrace_1* 

0
1*

0
1*
* 
Ш
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

Сtrace_0* 

Тtrace_0* 
Ф
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses* 
 
0
1
2
3*
 
0
1
2
3*
* 
Ш
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Юtrace_0* 

Яtrace_0* 
* 
Ф
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses* 

¶0*
 
0
1
2
3*
 
0
1
2
3*
* 
Ю
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*

ђtrace_0* 

≠trace_0* 
* 
Ф
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses* 

і0*
 
0
1
2
3*
 
0
1
2
3*
* 
Ю
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

Їtrace_0* 

їtrace_0* 
* 
Ф
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses* 

¬0*
* 
* 
* 
Ь
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses* 

»trace_0* 

…trace_0* 

0
1*

0
1*
* 
Ю
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses*

ѕtrace_0* 

–trace_0* 
Ф
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses* 
* 
* 
* 
Ь
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses* 

№trace_0* 

Ёtrace_0* 

0
1*

0
1*
* 
Ю
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses*

гtrace_0* 

дtrace_0* 
Ф
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses* 

0
 1*

0
 1*
* 
Ю
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 

!0
"1*

!0
"1*
* 
Ю
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses*

чtrace_0* 

шtrace_0* 
* 
* 
* 
Ю
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses*

юtrace_0* 

€trace_0* 
+
	?layer
А	variables
Бoutputs* 
-
	@layer
В	variables
Гoutputs*
-
	Alayer
Д	variables
Еoutputs*
-
	Blayer
Ж	variables
Зoutputs*
-
	Clayer
И	variables
Йoutputs*
+
	Dlayer
К	variables
Лoutputs* 
-
	Elayer
М	variables
Нoutputs*
+
	Flayer
О	variables
Пoutputs* 
-
	Glayer
Р	variables
Сoutputs*
-
	Hlayer
Т	variables
Уoutputs*
-
	Ilayer
Ф	variables
Хoutputs*
-
	Jlayer
Ц	variables
Чoutputs*

)0*

R0
S1
T2*
* 
* 
* 
* 
* 
* 
* 

#0
$1*

#0
$1*
* 
Ю
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
÷	variables
„trainable_variables
Ўregularization_losses
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
Ф
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
£__call__
+§&call_and_return_all_conditional_losses* 

%0
&1*

%0
&1*
* 
Ю
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
Ё	variables
ёtrainable_variables
яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*

™trace_0* 

Ђtrace_0* 

'0
(1*

'0
(1*
* 
Ю
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*

±trace_0* 

≤trace_0* 
-
	Rlayer
≥	variables
іoutputs*
-
	Slayer
µ	variables
ґoutputs*
-
	Tlayer
Ј	variables
Єoutputs*

єconcat
Їoutputs* 

п0
р1*

н	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

у0
ф1*

с	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ш0
щ1*

ц	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

э0
ю1*

ы	variables*
oi
VARIABLE_VALUE!vae/nyan_encoder/sampling/total_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE!vae/nyan_encoder/sampling/count_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*

Б0
В1*

€	variables*
mg
VARIABLE_VALUEvae/nyan_encoder/sampling/total4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEvae/nyan_encoder/sampling/count4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
y0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses* 
* 
* 
* 

Б0
¶1*
* 
* 
* 
* 
* 
* 
* 
* 
Ь
јnon_trainable_variables
Ѕlayers
¬metrics
 √layer_regularization_losses
ƒlayer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses* 
* 
* 
ђ
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses

kernel
bias*
* 

К0
і1*
* 
* 
* 
* 
* 
* 
* 
* 
Ь
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses* 
* 
* 
ђ
–	variables
—trainable_variables
“regularization_losses
”	keras_api
‘__call__
+’&call_and_return_all_conditional_losses

kernel
bias*
* 

У0
¬1*
* 
* 
* 
* 
* 
* 
* 
* 
Ь
÷non_trainable_variables
„layers
Ўmetrics
 ўlayer_regularization_losses
Џlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses* 
* 
* 
ђ
џ	variables
№trainable_variables
Ёregularization_losses
ё	keras_api
я__call__
+а&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
* 
* 
* 
* 
* 


°0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


Ѓ0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*
* 
#
^kl_loss
_kl_loss_beta*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


№0* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Я	variables
†trainable_variables
°regularization_losses
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
Ю
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
Ю
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
–	variables
—trainable_variables
“regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
Ю
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
џ	variables
№trainable_variables
Ёregularization_losses
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Аz
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/vae/nyan_encoder/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/vae/nyan_encoder/ecc_conv/root_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%Adam/vae/nyan_encoder/ecc_conv/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE'Adam/vae/nyan_encoder/ecc_conv_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE'Adam/vae/nyan_encoder/ecc_conv_2/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/dense_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/dense_2/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense_2/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE%Adam/vae/nyan_encoder/z_mean/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE#Adam/vae/nyan_encoder/z_mean/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/vae/nyan_encoder/z_log_var/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/z_log_var/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_3/kernel/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_3/bias/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_4/kernel/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_4/bias/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_5/kernel/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_5/bias/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/vae/nyan_encoder/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/vae/nyan_encoder/ecc_conv/root_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE%Adam/vae/nyan_encoder/ecc_conv/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE'Adam/vae/nyan_encoder/ecc_conv_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE'Adam/vae/nyan_encoder/ecc_conv_2/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/dense_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/dense_2/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_encoder/dense_2/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE%Adam/vae/nyan_encoder/z_mean/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE#Adam/vae/nyan_encoder/z_mean/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/vae/nyan_encoder/z_log_var/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_encoder/z_log_var/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_3/kernel/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_3/bias/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_4/kernel/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_4/bias/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/vae/nyan_decoder/dense_5/kernel/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE$Adam/vae/nyan_decoder/dense_5/bias/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!global_step_1/Read/ReadVariableOp1vae/nyan_encoder/dense/kernel/Read/ReadVariableOp/vae/nyan_encoder/dense/bias/Read/ReadVariableOp9vae/nyan_encoder/ecc_conv/root_kernel/Read/ReadVariableOp2vae/nyan_encoder/ecc_conv/bias/Read/ReadVariableOp<vae/nyan_encoder/ecc_conv/FGN_out/kernel/Read/ReadVariableOp:vae/nyan_encoder/ecc_conv/FGN_out/bias/Read/ReadVariableOp;vae/nyan_encoder/ecc_conv_1/root_kernel/Read/ReadVariableOp4vae/nyan_encoder/ecc_conv_1/bias/Read/ReadVariableOp>vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/Read/ReadVariableOp<vae/nyan_encoder/ecc_conv_1/FGN_out/bias/Read/ReadVariableOp;vae/nyan_encoder/ecc_conv_2/root_kernel/Read/ReadVariableOp4vae/nyan_encoder/ecc_conv_2/bias/Read/ReadVariableOp>vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/Read/ReadVariableOp<vae/nyan_encoder/ecc_conv_2/FGN_out/bias/Read/ReadVariableOp3vae/nyan_encoder/dense_1/kernel/Read/ReadVariableOp1vae/nyan_encoder/dense_1/bias/Read/ReadVariableOp3vae/nyan_encoder/dense_2/kernel/Read/ReadVariableOp1vae/nyan_encoder/dense_2/bias/Read/ReadVariableOp2vae/nyan_encoder/z_mean/kernel/Read/ReadVariableOp0vae/nyan_encoder/z_mean/bias/Read/ReadVariableOp5vae/nyan_encoder/z_log_var/kernel/Read/ReadVariableOp3vae/nyan_encoder/z_log_var/bias/Read/ReadVariableOp3vae/nyan_decoder/dense_3/kernel/Read/ReadVariableOp1vae/nyan_decoder/dense_3/bias/Read/ReadVariableOp3vae/nyan_decoder/dense_4/kernel/Read/ReadVariableOp1vae/nyan_decoder/dense_4/bias/Read/ReadVariableOp3vae/nyan_decoder/dense_5/kernel/Read/ReadVariableOp1vae/nyan_decoder/dense_5/bias/Read/ReadVariableOpglobal_step/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5vae/nyan_encoder/sampling/total_1/Read/ReadVariableOp5vae/nyan_encoder/sampling/count_1/Read/ReadVariableOp3vae/nyan_encoder/sampling/total/Read/ReadVariableOp3vae/nyan_encoder/sampling/count/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense/kernel/m/Read/ReadVariableOp6Adam/vae/nyan_encoder/dense/bias/m/Read/ReadVariableOp@Adam/vae/nyan_encoder/ecc_conv/root_kernel/m/Read/ReadVariableOp9Adam/vae/nyan_encoder/ecc_conv/bias/m/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m/Read/ReadVariableOpAAdam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m/Read/ReadVariableOpBAdam/vae/nyan_encoder/ecc_conv_1/root_kernel/m/Read/ReadVariableOp;Adam/vae/nyan_encoder/ecc_conv_1/bias/m/Read/ReadVariableOpEAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m/Read/ReadVariableOpBAdam/vae/nyan_encoder/ecc_conv_2/root_kernel/m/Read/ReadVariableOp;Adam/vae/nyan_encoder/ecc_conv_2/bias/m/Read/ReadVariableOpEAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m/Read/ReadVariableOp:Adam/vae/nyan_encoder/dense_1/kernel/m/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense_1/bias/m/Read/ReadVariableOp:Adam/vae/nyan_encoder/dense_2/kernel/m/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense_2/bias/m/Read/ReadVariableOp9Adam/vae/nyan_encoder/z_mean/kernel/m/Read/ReadVariableOp7Adam/vae/nyan_encoder/z_mean/bias/m/Read/ReadVariableOp<Adam/vae/nyan_encoder/z_log_var/kernel/m/Read/ReadVariableOp:Adam/vae/nyan_encoder/z_log_var/bias/m/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_3/kernel/m/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_3/bias/m/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_4/kernel/m/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_4/bias/m/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_5/kernel/m/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_5/bias/m/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense/kernel/v/Read/ReadVariableOp6Adam/vae/nyan_encoder/dense/bias/v/Read/ReadVariableOp@Adam/vae/nyan_encoder/ecc_conv/root_kernel/v/Read/ReadVariableOp9Adam/vae/nyan_encoder/ecc_conv/bias/v/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v/Read/ReadVariableOpAAdam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v/Read/ReadVariableOpBAdam/vae/nyan_encoder/ecc_conv_1/root_kernel/v/Read/ReadVariableOp;Adam/vae/nyan_encoder/ecc_conv_1/bias/v/Read/ReadVariableOpEAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v/Read/ReadVariableOpBAdam/vae/nyan_encoder/ecc_conv_2/root_kernel/v/Read/ReadVariableOp;Adam/vae/nyan_encoder/ecc_conv_2/bias/v/Read/ReadVariableOpEAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v/Read/ReadVariableOpCAdam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v/Read/ReadVariableOp:Adam/vae/nyan_encoder/dense_1/kernel/v/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense_1/bias/v/Read/ReadVariableOp:Adam/vae/nyan_encoder/dense_2/kernel/v/Read/ReadVariableOp8Adam/vae/nyan_encoder/dense_2/bias/v/Read/ReadVariableOp9Adam/vae/nyan_encoder/z_mean/kernel/v/Read/ReadVariableOp7Adam/vae/nyan_encoder/z_mean/bias/v/Read/ReadVariableOp<Adam/vae/nyan_encoder/z_log_var/kernel/v/Read/ReadVariableOp:Adam/vae/nyan_encoder/z_log_var/bias/v/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_3/kernel/v/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_3/bias/v/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_4/kernel/v/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_4/bias/v/Read/ReadVariableOp:Adam/vae/nyan_decoder/dense_5/kernel/v/Read/ReadVariableOp8Adam/vae/nyan_decoder/dense_5/bias/v/Read/ReadVariableOpConst*q
Tinj
h2f	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_save_7412169
Ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step_1vae/nyan_encoder/dense/kernelvae/nyan_encoder/dense/bias%vae/nyan_encoder/ecc_conv/root_kernelvae/nyan_encoder/ecc_conv/bias(vae/nyan_encoder/ecc_conv/FGN_out/kernel&vae/nyan_encoder/ecc_conv/FGN_out/bias'vae/nyan_encoder/ecc_conv_1/root_kernel vae/nyan_encoder/ecc_conv_1/bias*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel(vae/nyan_encoder/ecc_conv_1/FGN_out/bias'vae/nyan_encoder/ecc_conv_2/root_kernel vae/nyan_encoder/ecc_conv_2/bias*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel(vae/nyan_encoder/ecc_conv_2/FGN_out/biasvae/nyan_encoder/dense_1/kernelvae/nyan_encoder/dense_1/biasvae/nyan_encoder/dense_2/kernelvae/nyan_encoder/dense_2/biasvae/nyan_encoder/z_mean/kernelvae/nyan_encoder/z_mean/bias!vae/nyan_encoder/z_log_var/kernelvae/nyan_encoder/z_log_var/biasvae/nyan_decoder/dense_3/kernelvae/nyan_decoder/dense_3/biasvae/nyan_decoder/dense_4/kernelvae/nyan_decoder/dense_4/biasvae/nyan_decoder/dense_5/kernelvae/nyan_decoder/dense_5/biasglobal_step	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotal_2count_2total_1count_1totalcount!vae/nyan_encoder/sampling/total_1!vae/nyan_encoder/sampling/count_1vae/nyan_encoder/sampling/totalvae/nyan_encoder/sampling/count$Adam/vae/nyan_encoder/dense/kernel/m"Adam/vae/nyan_encoder/dense/bias/m,Adam/vae/nyan_encoder/ecc_conv/root_kernel/m%Adam/vae/nyan_encoder/ecc_conv/bias/m/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/m'Adam/vae/nyan_encoder/ecc_conv_1/bias/m1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/m'Adam/vae/nyan_encoder/ecc_conv_2/bias/m1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m&Adam/vae/nyan_encoder/dense_1/kernel/m$Adam/vae/nyan_encoder/dense_1/bias/m&Adam/vae/nyan_encoder/dense_2/kernel/m$Adam/vae/nyan_encoder/dense_2/bias/m%Adam/vae/nyan_encoder/z_mean/kernel/m#Adam/vae/nyan_encoder/z_mean/bias/m(Adam/vae/nyan_encoder/z_log_var/kernel/m&Adam/vae/nyan_encoder/z_log_var/bias/m&Adam/vae/nyan_decoder/dense_3/kernel/m$Adam/vae/nyan_decoder/dense_3/bias/m&Adam/vae/nyan_decoder/dense_4/kernel/m$Adam/vae/nyan_decoder/dense_4/bias/m&Adam/vae/nyan_decoder/dense_5/kernel/m$Adam/vae/nyan_decoder/dense_5/bias/m$Adam/vae/nyan_encoder/dense/kernel/v"Adam/vae/nyan_encoder/dense/bias/v,Adam/vae/nyan_encoder/ecc_conv/root_kernel/v%Adam/vae/nyan_encoder/ecc_conv/bias/v/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/v'Adam/vae/nyan_encoder/ecc_conv_1/bias/v1Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/v'Adam/vae/nyan_encoder/ecc_conv_2/bias/v1Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v&Adam/vae/nyan_encoder/dense_1/kernel/v$Adam/vae/nyan_encoder/dense_1/bias/v&Adam/vae/nyan_encoder/dense_2/kernel/v$Adam/vae/nyan_encoder/dense_2/bias/v%Adam/vae/nyan_encoder/z_mean/kernel/v#Adam/vae/nyan_encoder/z_mean/bias/v(Adam/vae/nyan_encoder/z_log_var/kernel/v&Adam/vae/nyan_encoder/z_log_var/bias/v&Adam/vae/nyan_decoder/dense_3/kernel/v$Adam/vae/nyan_decoder/dense_3/bias/v&Adam/vae/nyan_decoder/dense_4/kernel/v$Adam/vae/nyan_decoder/dense_4/bias/v&Adam/vae/nyan_decoder/dense_5/kernel/v$Adam/vae/nyan_decoder/dense_5/bias/v*p
Tini
g2e*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__traced_restore_7412479Ѕп$
Х
€
%__inference_vae_layer_call_fn_7408880
input_1
input_2	
input_3
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26:	@А

unknown_27:	А

unknown_28:
АІ

unknown_29:	І

unknown_30:
АЌ

unknown_31:	Ќ
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:€€€€€€€€€ф: *?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vae_layer_call_and_return_conditional_losses_7408810p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
•Щ
‘I
#__inference__traced_restore_7412479
file_prefix(
assignvariableop_global_step_1: B
0assignvariableop_1_vae_nyan_encoder_dense_kernel:<
.assignvariableop_2_vae_nyan_encoder_dense_bias:J
8assignvariableop_3_vae_nyan_encoder_ecc_conv_root_kernel: ?
1assignvariableop_4_vae_nyan_encoder_ecc_conv_bias: N
;assignvariableop_5_vae_nyan_encoder_ecc_conv_fgn_out_kernel:	АH
9assignvariableop_6_vae_nyan_encoder_ecc_conv_fgn_out_bias:	АL
:assignvariableop_7_vae_nyan_encoder_ecc_conv_1_root_kernel:  A
3assignvariableop_8_vae_nyan_encoder_ecc_conv_1_bias: P
=assignvariableop_9_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel:	АK
<assignvariableop_10_vae_nyan_encoder_ecc_conv_1_fgn_out_bias:	АM
;assignvariableop_11_vae_nyan_encoder_ecc_conv_2_root_kernel:  B
4assignvariableop_12_vae_nyan_encoder_ecc_conv_2_bias: Q
>assignvariableop_13_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel:	АK
<assignvariableop_14_vae_nyan_encoder_ecc_conv_2_fgn_out_bias:	АF
3assignvariableop_15_vae_nyan_encoder_dense_1_kernel:	 А@
1assignvariableop_16_vae_nyan_encoder_dense_1_bias:	АG
3assignvariableop_17_vae_nyan_encoder_dense_2_kernel:
АА@
1assignvariableop_18_vae_nyan_encoder_dense_2_bias:	АE
2assignvariableop_19_vae_nyan_encoder_z_mean_kernel:	А@>
0assignvariableop_20_vae_nyan_encoder_z_mean_bias:@H
5assignvariableop_21_vae_nyan_encoder_z_log_var_kernel:	А@A
3assignvariableop_22_vae_nyan_encoder_z_log_var_bias:@F
3assignvariableop_23_vae_nyan_decoder_dense_3_kernel:	@А@
1assignvariableop_24_vae_nyan_decoder_dense_3_bias:	АG
3assignvariableop_25_vae_nyan_decoder_dense_4_kernel:
АІ@
1assignvariableop_26_vae_nyan_decoder_dense_4_bias:	ІG
3assignvariableop_27_vae_nyan_decoder_dense_5_kernel:
АЌ@
1assignvariableop_28_vae_nyan_decoder_dense_5_bias:	Ќ)
assignvariableop_29_global_step: '
assignvariableop_30_adam_iter:	 )
assignvariableop_31_adam_beta_1: )
assignvariableop_32_adam_beta_2: (
assignvariableop_33_adam_decay: %
assignvariableop_34_total_2: %
assignvariableop_35_count_2: %
assignvariableop_36_total_1: %
assignvariableop_37_count_1: #
assignvariableop_38_total: #
assignvariableop_39_count: ?
5assignvariableop_40_vae_nyan_encoder_sampling_total_1: ?
5assignvariableop_41_vae_nyan_encoder_sampling_count_1: =
3assignvariableop_42_vae_nyan_encoder_sampling_total: =
3assignvariableop_43_vae_nyan_encoder_sampling_count: J
8assignvariableop_44_adam_vae_nyan_encoder_dense_kernel_m:D
6assignvariableop_45_adam_vae_nyan_encoder_dense_bias_m:R
@assignvariableop_46_adam_vae_nyan_encoder_ecc_conv_root_kernel_m: G
9assignvariableop_47_adam_vae_nyan_encoder_ecc_conv_bias_m: V
Cassignvariableop_48_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_m:	АP
Aassignvariableop_49_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_m:	АT
Bassignvariableop_50_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_m:  I
;assignvariableop_51_adam_vae_nyan_encoder_ecc_conv_1_bias_m: X
Eassignvariableop_52_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_m:	АR
Cassignvariableop_53_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_m:	АT
Bassignvariableop_54_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_m:  I
;assignvariableop_55_adam_vae_nyan_encoder_ecc_conv_2_bias_m: X
Eassignvariableop_56_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_m:	АR
Cassignvariableop_57_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_m:	АM
:assignvariableop_58_adam_vae_nyan_encoder_dense_1_kernel_m:	 АG
8assignvariableop_59_adam_vae_nyan_encoder_dense_1_bias_m:	АN
:assignvariableop_60_adam_vae_nyan_encoder_dense_2_kernel_m:
ААG
8assignvariableop_61_adam_vae_nyan_encoder_dense_2_bias_m:	АL
9assignvariableop_62_adam_vae_nyan_encoder_z_mean_kernel_m:	А@E
7assignvariableop_63_adam_vae_nyan_encoder_z_mean_bias_m:@O
<assignvariableop_64_adam_vae_nyan_encoder_z_log_var_kernel_m:	А@H
:assignvariableop_65_adam_vae_nyan_encoder_z_log_var_bias_m:@M
:assignvariableop_66_adam_vae_nyan_decoder_dense_3_kernel_m:	@АG
8assignvariableop_67_adam_vae_nyan_decoder_dense_3_bias_m:	АN
:assignvariableop_68_adam_vae_nyan_decoder_dense_4_kernel_m:
АІG
8assignvariableop_69_adam_vae_nyan_decoder_dense_4_bias_m:	ІN
:assignvariableop_70_adam_vae_nyan_decoder_dense_5_kernel_m:
АЌG
8assignvariableop_71_adam_vae_nyan_decoder_dense_5_bias_m:	ЌJ
8assignvariableop_72_adam_vae_nyan_encoder_dense_kernel_v:D
6assignvariableop_73_adam_vae_nyan_encoder_dense_bias_v:R
@assignvariableop_74_adam_vae_nyan_encoder_ecc_conv_root_kernel_v: G
9assignvariableop_75_adam_vae_nyan_encoder_ecc_conv_bias_v: V
Cassignvariableop_76_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_v:	АP
Aassignvariableop_77_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_v:	АT
Bassignvariableop_78_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_v:  I
;assignvariableop_79_adam_vae_nyan_encoder_ecc_conv_1_bias_v: X
Eassignvariableop_80_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_v:	АR
Cassignvariableop_81_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_v:	АT
Bassignvariableop_82_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_v:  I
;assignvariableop_83_adam_vae_nyan_encoder_ecc_conv_2_bias_v: X
Eassignvariableop_84_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_v:	АR
Cassignvariableop_85_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_v:	АM
:assignvariableop_86_adam_vae_nyan_encoder_dense_1_kernel_v:	 АG
8assignvariableop_87_adam_vae_nyan_encoder_dense_1_bias_v:	АN
:assignvariableop_88_adam_vae_nyan_encoder_dense_2_kernel_v:
ААG
8assignvariableop_89_adam_vae_nyan_encoder_dense_2_bias_v:	АL
9assignvariableop_90_adam_vae_nyan_encoder_z_mean_kernel_v:	А@E
7assignvariableop_91_adam_vae_nyan_encoder_z_mean_bias_v:@O
<assignvariableop_92_adam_vae_nyan_encoder_z_log_var_kernel_v:	А@H
:assignvariableop_93_adam_vae_nyan_encoder_z_log_var_bias_v:@M
:assignvariableop_94_adam_vae_nyan_decoder_dense_3_kernel_v:	@АG
8assignvariableop_95_adam_vae_nyan_decoder_dense_3_bias_v:	АN
:assignvariableop_96_adam_vae_nyan_decoder_dense_4_kernel_v:
АІG
8assignvariableop_97_adam_vae_nyan_decoder_dense_4_bias_v:	ІN
:assignvariableop_98_adam_vae_nyan_decoder_dense_5_kernel_v:
АЌG
8assignvariableop_99_adam_vae_nyan_decoder_dense_5_bias_v:	Ќ
identity_101ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99Ь.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*¬-
valueЄ-Bµ-eB!epochs/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHљ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*я
value’B“eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ъ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*™
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*s
dtypesi
g2e	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_global_step_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_1AssignVariableOp0assignvariableop_1_vae_nyan_encoder_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_2AssignVariableOp.assignvariableop_2_vae_nyan_encoder_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_3AssignVariableOp8assignvariableop_3_vae_nyan_encoder_ecc_conv_root_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_4AssignVariableOp1assignvariableop_4_vae_nyan_encoder_ecc_conv_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_5AssignVariableOp;assignvariableop_5_vae_nyan_encoder_ecc_conv_fgn_out_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_6AssignVariableOp9assignvariableop_6_vae_nyan_encoder_ecc_conv_fgn_out_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_7AssignVariableOp:assignvariableop_7_vae_nyan_encoder_ecc_conv_1_root_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_8AssignVariableOp3assignvariableop_8_vae_nyan_encoder_ecc_conv_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_9AssignVariableOp=assignvariableop_9_vae_nyan_encoder_ecc_conv_1_fgn_out_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_10AssignVariableOp<assignvariableop_10_vae_nyan_encoder_ecc_conv_1_fgn_out_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_11AssignVariableOp;assignvariableop_11_vae_nyan_encoder_ecc_conv_2_root_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_12AssignVariableOp4assignvariableop_12_vae_nyan_encoder_ecc_conv_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_13AssignVariableOp>assignvariableop_13_vae_nyan_encoder_ecc_conv_2_fgn_out_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_14AssignVariableOp<assignvariableop_14_vae_nyan_encoder_ecc_conv_2_fgn_out_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_15AssignVariableOp3assignvariableop_15_vae_nyan_encoder_dense_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_16AssignVariableOp1assignvariableop_16_vae_nyan_encoder_dense_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_17AssignVariableOp3assignvariableop_17_vae_nyan_encoder_dense_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_18AssignVariableOp1assignvariableop_18_vae_nyan_encoder_dense_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_19AssignVariableOp2assignvariableop_19_vae_nyan_encoder_z_mean_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_20AssignVariableOp0assignvariableop_20_vae_nyan_encoder_z_mean_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_21AssignVariableOp5assignvariableop_21_vae_nyan_encoder_z_log_var_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp3assignvariableop_22_vae_nyan_encoder_z_log_var_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_23AssignVariableOp3assignvariableop_23_vae_nyan_decoder_dense_3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_24AssignVariableOp1assignvariableop_24_vae_nyan_decoder_dense_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_25AssignVariableOp3assignvariableop_25_vae_nyan_decoder_dense_4_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_26AssignVariableOp1assignvariableop_26_vae_nyan_decoder_dense_4_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_27AssignVariableOp3assignvariableop_27_vae_nyan_decoder_dense_5_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_28AssignVariableOp1assignvariableop_28_vae_nyan_decoder_dense_5_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_29AssignVariableOpassignvariableop_29_global_stepIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_2Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_40AssignVariableOp5assignvariableop_40_vae_nyan_encoder_sampling_total_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_41AssignVariableOp5assignvariableop_41_vae_nyan_encoder_sampling_count_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_42AssignVariableOp3assignvariableop_42_vae_nyan_encoder_sampling_totalIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_43AssignVariableOp3assignvariableop_43_vae_nyan_encoder_sampling_countIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_44AssignVariableOp8assignvariableop_44_adam_vae_nyan_encoder_dense_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_vae_nyan_encoder_dense_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_vae_nyan_encoder_ecc_conv_root_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_47AssignVariableOp9assignvariableop_47_adam_vae_nyan_encoder_ecc_conv_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_48AssignVariableOpCassignvariableop_48_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_49AssignVariableOpAassignvariableop_49_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_50AssignVariableOpBassignvariableop_50_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_51AssignVariableOp;assignvariableop_51_adam_vae_nyan_encoder_ecc_conv_1_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_52AssignVariableOpEassignvariableop_52_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_53AssignVariableOpCassignvariableop_53_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_54AssignVariableOpBassignvariableop_54_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_55AssignVariableOp;assignvariableop_55_adam_vae_nyan_encoder_ecc_conv_2_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_56AssignVariableOpEassignvariableop_56_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_57AssignVariableOpCassignvariableop_57_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_58AssignVariableOp:assignvariableop_58_adam_vae_nyan_encoder_dense_1_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_vae_nyan_encoder_dense_1_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_60AssignVariableOp:assignvariableop_60_adam_vae_nyan_encoder_dense_2_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_61AssignVariableOp8assignvariableop_61_adam_vae_nyan_encoder_dense_2_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_62AssignVariableOp9assignvariableop_62_adam_vae_nyan_encoder_z_mean_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_vae_nyan_encoder_z_mean_bias_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_64AssignVariableOp<assignvariableop_64_adam_vae_nyan_encoder_z_log_var_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_65AssignVariableOp:assignvariableop_65_adam_vae_nyan_encoder_z_log_var_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_vae_nyan_decoder_dense_3_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_vae_nyan_decoder_dense_3_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_68AssignVariableOp:assignvariableop_68_adam_vae_nyan_decoder_dense_4_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_vae_nyan_decoder_dense_4_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_70AssignVariableOp:assignvariableop_70_adam_vae_nyan_decoder_dense_5_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_vae_nyan_decoder_dense_5_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_72AssignVariableOp8assignvariableop_72_adam_vae_nyan_encoder_dense_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_vae_nyan_encoder_dense_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_74AssignVariableOp@assignvariableop_74_adam_vae_nyan_encoder_ecc_conv_root_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_75AssignVariableOp9assignvariableop_75_adam_vae_nyan_encoder_ecc_conv_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_76AssignVariableOpCassignvariableop_76_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_77AssignVariableOpAassignvariableop_77_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_78AssignVariableOpBassignvariableop_78_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_79AssignVariableOp;assignvariableop_79_adam_vae_nyan_encoder_ecc_conv_1_bias_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_80AssignVariableOpEassignvariableop_80_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_81AssignVariableOpCassignvariableop_81_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_82AssignVariableOpBassignvariableop_82_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:ђ
AssignVariableOp_83AssignVariableOp;assignvariableop_83_adam_vae_nyan_encoder_ecc_conv_2_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_84AssignVariableOpEassignvariableop_84_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_85AssignVariableOpCassignvariableop_85_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_86AssignVariableOp:assignvariableop_86_adam_vae_nyan_encoder_dense_1_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_vae_nyan_encoder_dense_1_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_88AssignVariableOp:assignvariableop_88_adam_vae_nyan_encoder_dense_2_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_89AssignVariableOp8assignvariableop_89_adam_vae_nyan_encoder_dense_2_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_90AssignVariableOp9assignvariableop_90_adam_vae_nyan_encoder_z_mean_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_vae_nyan_encoder_z_mean_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_92AssignVariableOp<assignvariableop_92_adam_vae_nyan_encoder_z_log_var_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_93AssignVariableOp:assignvariableop_93_adam_vae_nyan_encoder_z_log_var_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_94AssignVariableOp:assignvariableop_94_adam_vae_nyan_decoder_dense_3_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_vae_nyan_decoder_dense_3_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_96AssignVariableOp:assignvariableop_96_adam_vae_nyan_decoder_dense_4_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_97AssignVariableOp8assignvariableop_97_adam_vae_nyan_decoder_dense_4_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_98AssignVariableOp:assignvariableop_98_adam_vae_nyan_decoder_dense_5_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_vae_nyan_decoder_dense_5_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 и
Identity_100Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_101IdentityIdentity_100:output:0^NoOp_1*
T0*
_output_shapes
: ‘
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_101Identity_101:output:0*я
_input_shapesЌ
 : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
љL
Џ
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590

inputs
inputs_1	
inputs_2
mask<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< Е
einsum/EinsumEinsummul:z:0inputs*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd=
Shape_2Shapeinputs*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   h
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

: j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Z
mul_1MulBiasAdd:output:0mask*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_1/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:QM
+
_output_shapes
:€€€€€€€€€<

_user_specified_namemask
ќ
ы
,__inference_ecc_conv_1_layer_call_fn_7411411
inputs_0
inputs_1	
inputs_2

mask_0
unknown:	А
	unknown_0:	А
	unknown_1:  
	unknown_2: 
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2mask_0unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€< 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
Б	
П
.__inference_nyan_decoder_layer_call_fn_7411206

inputs
unknown:	@А
	unknown_0:	А
	unknown_1:
АІ
	unknown_2:	І
	unknown_3:
АЌ
	unknown_4:	Ќ
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Я»
£5
 __inference__traced_save_7412169
file_prefix,
(savev2_global_step_1_read_readvariableop<
8savev2_vae_nyan_encoder_dense_kernel_read_readvariableop:
6savev2_vae_nyan_encoder_dense_bias_read_readvariableopD
@savev2_vae_nyan_encoder_ecc_conv_root_kernel_read_readvariableop=
9savev2_vae_nyan_encoder_ecc_conv_bias_read_readvariableopG
Csavev2_vae_nyan_encoder_ecc_conv_fgn_out_kernel_read_readvariableopE
Asavev2_vae_nyan_encoder_ecc_conv_fgn_out_bias_read_readvariableopF
Bsavev2_vae_nyan_encoder_ecc_conv_1_root_kernel_read_readvariableop?
;savev2_vae_nyan_encoder_ecc_conv_1_bias_read_readvariableopI
Esavev2_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_read_readvariableopG
Csavev2_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_read_readvariableopF
Bsavev2_vae_nyan_encoder_ecc_conv_2_root_kernel_read_readvariableop?
;savev2_vae_nyan_encoder_ecc_conv_2_bias_read_readvariableopI
Esavev2_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_read_readvariableopG
Csavev2_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_read_readvariableop>
:savev2_vae_nyan_encoder_dense_1_kernel_read_readvariableop<
8savev2_vae_nyan_encoder_dense_1_bias_read_readvariableop>
:savev2_vae_nyan_encoder_dense_2_kernel_read_readvariableop<
8savev2_vae_nyan_encoder_dense_2_bias_read_readvariableop=
9savev2_vae_nyan_encoder_z_mean_kernel_read_readvariableop;
7savev2_vae_nyan_encoder_z_mean_bias_read_readvariableop@
<savev2_vae_nyan_encoder_z_log_var_kernel_read_readvariableop>
:savev2_vae_nyan_encoder_z_log_var_bias_read_readvariableop>
:savev2_vae_nyan_decoder_dense_3_kernel_read_readvariableop<
8savev2_vae_nyan_decoder_dense_3_bias_read_readvariableop>
:savev2_vae_nyan_decoder_dense_4_kernel_read_readvariableop<
8savev2_vae_nyan_decoder_dense_4_bias_read_readvariableop>
:savev2_vae_nyan_decoder_dense_5_kernel_read_readvariableop<
8savev2_vae_nyan_decoder_dense_5_bias_read_readvariableop*
&savev2_global_step_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_vae_nyan_encoder_sampling_total_1_read_readvariableop@
<savev2_vae_nyan_encoder_sampling_count_1_read_readvariableop>
:savev2_vae_nyan_encoder_sampling_total_read_readvariableop>
:savev2_vae_nyan_encoder_sampling_count_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_kernel_m_read_readvariableopA
=savev2_adam_vae_nyan_encoder_dense_bias_m_read_readvariableopK
Gsavev2_adam_vae_nyan_encoder_ecc_conv_root_kernel_m_read_readvariableopD
@savev2_adam_vae_nyan_encoder_ecc_conv_bias_m_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_m_read_readvariableopL
Hsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_m_read_readvariableopM
Isavev2_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_m_read_readvariableopF
Bsavev2_adam_vae_nyan_encoder_ecc_conv_1_bias_m_read_readvariableopP
Lsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_m_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_m_read_readvariableopM
Isavev2_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_m_read_readvariableopF
Bsavev2_adam_vae_nyan_encoder_ecc_conv_2_bias_m_read_readvariableopP
Lsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_m_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_m_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_dense_1_kernel_m_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_1_bias_m_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_dense_2_kernel_m_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_2_bias_m_read_readvariableopD
@savev2_adam_vae_nyan_encoder_z_mean_kernel_m_read_readvariableopB
>savev2_adam_vae_nyan_encoder_z_mean_bias_m_read_readvariableopG
Csavev2_adam_vae_nyan_encoder_z_log_var_kernel_m_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_z_log_var_bias_m_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_3_kernel_m_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_3_bias_m_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_4_kernel_m_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_4_bias_m_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_5_kernel_m_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_5_bias_m_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_kernel_v_read_readvariableopA
=savev2_adam_vae_nyan_encoder_dense_bias_v_read_readvariableopK
Gsavev2_adam_vae_nyan_encoder_ecc_conv_root_kernel_v_read_readvariableopD
@savev2_adam_vae_nyan_encoder_ecc_conv_bias_v_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_v_read_readvariableopL
Hsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_v_read_readvariableopM
Isavev2_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_v_read_readvariableopF
Bsavev2_adam_vae_nyan_encoder_ecc_conv_1_bias_v_read_readvariableopP
Lsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_v_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_v_read_readvariableopM
Isavev2_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_v_read_readvariableopF
Bsavev2_adam_vae_nyan_encoder_ecc_conv_2_bias_v_read_readvariableopP
Lsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_v_read_readvariableopN
Jsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_v_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_dense_1_kernel_v_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_1_bias_v_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_dense_2_kernel_v_read_readvariableopC
?savev2_adam_vae_nyan_encoder_dense_2_bias_v_read_readvariableopD
@savev2_adam_vae_nyan_encoder_z_mean_kernel_v_read_readvariableopB
>savev2_adam_vae_nyan_encoder_z_mean_bias_v_read_readvariableopG
Csavev2_adam_vae_nyan_encoder_z_log_var_kernel_v_read_readvariableopE
Asavev2_adam_vae_nyan_encoder_z_log_var_bias_v_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_3_kernel_v_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_3_bias_v_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_4_kernel_v_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_4_bias_v_read_readvariableopE
Asavev2_adam_vae_nyan_decoder_dense_5_kernel_v_read_readvariableopC
?savev2_adam_vae_nyan_decoder_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Щ.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*¬-
valueЄ-Bµ-eB!epochs/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:e*
dtype0*я
value’B“eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ї3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_global_step_1_read_readvariableop8savev2_vae_nyan_encoder_dense_kernel_read_readvariableop6savev2_vae_nyan_encoder_dense_bias_read_readvariableop@savev2_vae_nyan_encoder_ecc_conv_root_kernel_read_readvariableop9savev2_vae_nyan_encoder_ecc_conv_bias_read_readvariableopCsavev2_vae_nyan_encoder_ecc_conv_fgn_out_kernel_read_readvariableopAsavev2_vae_nyan_encoder_ecc_conv_fgn_out_bias_read_readvariableopBsavev2_vae_nyan_encoder_ecc_conv_1_root_kernel_read_readvariableop;savev2_vae_nyan_encoder_ecc_conv_1_bias_read_readvariableopEsavev2_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_read_readvariableopCsavev2_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_read_readvariableopBsavev2_vae_nyan_encoder_ecc_conv_2_root_kernel_read_readvariableop;savev2_vae_nyan_encoder_ecc_conv_2_bias_read_readvariableopEsavev2_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_read_readvariableopCsavev2_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_read_readvariableop:savev2_vae_nyan_encoder_dense_1_kernel_read_readvariableop8savev2_vae_nyan_encoder_dense_1_bias_read_readvariableop:savev2_vae_nyan_encoder_dense_2_kernel_read_readvariableop8savev2_vae_nyan_encoder_dense_2_bias_read_readvariableop9savev2_vae_nyan_encoder_z_mean_kernel_read_readvariableop7savev2_vae_nyan_encoder_z_mean_bias_read_readvariableop<savev2_vae_nyan_encoder_z_log_var_kernel_read_readvariableop:savev2_vae_nyan_encoder_z_log_var_bias_read_readvariableop:savev2_vae_nyan_decoder_dense_3_kernel_read_readvariableop8savev2_vae_nyan_decoder_dense_3_bias_read_readvariableop:savev2_vae_nyan_decoder_dense_4_kernel_read_readvariableop8savev2_vae_nyan_decoder_dense_4_bias_read_readvariableop:savev2_vae_nyan_decoder_dense_5_kernel_read_readvariableop8savev2_vae_nyan_decoder_dense_5_bias_read_readvariableop&savev2_global_step_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_vae_nyan_encoder_sampling_total_1_read_readvariableop<savev2_vae_nyan_encoder_sampling_count_1_read_readvariableop:savev2_vae_nyan_encoder_sampling_total_read_readvariableop:savev2_vae_nyan_encoder_sampling_count_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_kernel_m_read_readvariableop=savev2_adam_vae_nyan_encoder_dense_bias_m_read_readvariableopGsavev2_adam_vae_nyan_encoder_ecc_conv_root_kernel_m_read_readvariableop@savev2_adam_vae_nyan_encoder_ecc_conv_bias_m_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_m_read_readvariableopHsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_m_read_readvariableopIsavev2_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_m_read_readvariableopBsavev2_adam_vae_nyan_encoder_ecc_conv_1_bias_m_read_readvariableopLsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_m_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_m_read_readvariableopIsavev2_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_m_read_readvariableopBsavev2_adam_vae_nyan_encoder_ecc_conv_2_bias_m_read_readvariableopLsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_m_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_m_read_readvariableopAsavev2_adam_vae_nyan_encoder_dense_1_kernel_m_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_1_bias_m_read_readvariableopAsavev2_adam_vae_nyan_encoder_dense_2_kernel_m_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_2_bias_m_read_readvariableop@savev2_adam_vae_nyan_encoder_z_mean_kernel_m_read_readvariableop>savev2_adam_vae_nyan_encoder_z_mean_bias_m_read_readvariableopCsavev2_adam_vae_nyan_encoder_z_log_var_kernel_m_read_readvariableopAsavev2_adam_vae_nyan_encoder_z_log_var_bias_m_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_3_kernel_m_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_3_bias_m_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_4_kernel_m_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_4_bias_m_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_5_kernel_m_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_5_bias_m_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_kernel_v_read_readvariableop=savev2_adam_vae_nyan_encoder_dense_bias_v_read_readvariableopGsavev2_adam_vae_nyan_encoder_ecc_conv_root_kernel_v_read_readvariableop@savev2_adam_vae_nyan_encoder_ecc_conv_bias_v_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_kernel_v_read_readvariableopHsavev2_adam_vae_nyan_encoder_ecc_conv_fgn_out_bias_v_read_readvariableopIsavev2_adam_vae_nyan_encoder_ecc_conv_1_root_kernel_v_read_readvariableopBsavev2_adam_vae_nyan_encoder_ecc_conv_1_bias_v_read_readvariableopLsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_kernel_v_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_1_fgn_out_bias_v_read_readvariableopIsavev2_adam_vae_nyan_encoder_ecc_conv_2_root_kernel_v_read_readvariableopBsavev2_adam_vae_nyan_encoder_ecc_conv_2_bias_v_read_readvariableopLsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_kernel_v_read_readvariableopJsavev2_adam_vae_nyan_encoder_ecc_conv_2_fgn_out_bias_v_read_readvariableopAsavev2_adam_vae_nyan_encoder_dense_1_kernel_v_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_1_bias_v_read_readvariableopAsavev2_adam_vae_nyan_encoder_dense_2_kernel_v_read_readvariableop?savev2_adam_vae_nyan_encoder_dense_2_bias_v_read_readvariableop@savev2_adam_vae_nyan_encoder_z_mean_kernel_v_read_readvariableop>savev2_adam_vae_nyan_encoder_z_mean_bias_v_read_readvariableopCsavev2_adam_vae_nyan_encoder_z_log_var_kernel_v_read_readvariableopAsavev2_adam_vae_nyan_encoder_z_log_var_bias_v_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_3_kernel_v_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_3_bias_v_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_4_kernel_v_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_4_bias_v_read_readvariableopAsavev2_adam_vae_nyan_decoder_dense_5_kernel_v_read_readvariableop?savev2_adam_vae_nyan_decoder_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *s
dtypesi
g2e	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ш
_input_shapesЖ
Г: : ::: : :	А:А:  : :	А:А:  : :	А:А:	 А:А:
АА:А:	А@:@:	А@:@:	@А:А:
АІ:І:
АЌ:Ќ: : : : : : : : : : : : : : : ::: : :	А:А:  : :	А:А:  : :	А:А:	 А:А:
АА:А:	А@:@:	А@:@:	@А:А:
АІ:І:
АЌ:Ќ::: : :	А:А:  : :	А:А:  : :	А:А:	 А:А:
АА:А:	А@:@:	А@:@:	@А:А:
АІ:І:
АЌ:Ќ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :%!

_output_shapes
:	А:!

_output_shapes	
:А:$ 

_output_shapes

:  : 	

_output_shapes
: :%
!

_output_shapes
:	А:!

_output_shapes	
:А:$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	А:!

_output_shapes	
:А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:%!

_output_shapes
:	А@: 

_output_shapes
:@:%!

_output_shapes
:	@А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АІ:!

_output_shapes	
:І:&"
 
_output_shapes
:
АЌ:!

_output_shapes	
:Ќ:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :$- 

_output_shapes

:: .

_output_shapes
::$/ 

_output_shapes

: : 0

_output_shapes
: :%1!

_output_shapes
:	А:!2

_output_shapes	
:А:$3 

_output_shapes

:  : 4

_output_shapes
: :%5!

_output_shapes
:	А:!6

_output_shapes	
:А:$7 

_output_shapes

:  : 8

_output_shapes
: :%9!

_output_shapes
:	А:!:

_output_shapes	
:А:%;!

_output_shapes
:	 А:!<

_output_shapes	
:А:&="
 
_output_shapes
:
АА:!>

_output_shapes	
:А:%?!

_output_shapes
:	А@: @

_output_shapes
:@:%A!

_output_shapes
:	А@: B

_output_shapes
:@:%C!

_output_shapes
:	@А:!D

_output_shapes	
:А:&E"
 
_output_shapes
:
АІ:!F

_output_shapes	
:І:&G"
 
_output_shapes
:
АЌ:!H

_output_shapes	
:Ќ:$I 

_output_shapes

:: J

_output_shapes
::$K 

_output_shapes

: : L

_output_shapes
: :%M!

_output_shapes
:	А:!N

_output_shapes	
:А:$O 

_output_shapes

:  : P

_output_shapes
: :%Q!

_output_shapes
:	А:!R

_output_shapes	
:А:$S 

_output_shapes

:  : T

_output_shapes
: :%U!

_output_shapes
:	А:!V

_output_shapes	
:А:%W!

_output_shapes
:	 А:!X

_output_shapes	
:А:&Y"
 
_output_shapes
:
АА:!Z

_output_shapes	
:А:%[!

_output_shapes
:	А@: \

_output_shapes
:@:%]!

_output_shapes
:	А@: ^

_output_shapes
:@:%_!

_output_shapes
:	@А:!`

_output_shapes	
:А:&a"
 
_output_shapes
:
АІ:!b

_output_shapes	
:І:&c"
 
_output_shapes
:
АЌ:!d

_output_shapes	
:Ќ:e

_output_shapes
: 
§
E
)__inference_flatten_layer_call_fn_7411625

inputs
identity≥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7407813a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
“
h
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7411600

inputs
identity`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€< :S O
+
_output_shapes
:€€€€€€€€€< 
 
_user_specified_nameinputs
у
ґ
@__inference_vae_layer_call_and_return_conditional_losses_7409034

inputs
inputs_1	
inputs_2&
nyan_encoder_7408963:"
nyan_encoder_7408965:'
nyan_encoder_7408967:	А#
nyan_encoder_7408969:	А&
nyan_encoder_7408971: "
nyan_encoder_7408973: '
nyan_encoder_7408975:	А#
nyan_encoder_7408977:	А&
nyan_encoder_7408979:  "
nyan_encoder_7408981: '
nyan_encoder_7408983:	А#
nyan_encoder_7408985:	А&
nyan_encoder_7408987:  "
nyan_encoder_7408989: '
nyan_encoder_7408991:	 А#
nyan_encoder_7408993:	А(
nyan_encoder_7408995:
АА#
nyan_encoder_7408997:	А'
nyan_encoder_7408999:	А@"
nyan_encoder_7409001:@'
nyan_encoder_7409003:	А@"
nyan_encoder_7409005:@
nyan_encoder_7409007: 
nyan_encoder_7409009: 
nyan_encoder_7409011: 
nyan_encoder_7409013: 
nyan_encoder_7409015: '
nyan_decoder_7409019:	@А#
nyan_decoder_7409021:	А(
nyan_decoder_7409023:
АІ#
nyan_decoder_7409025:	І(
nyan_decoder_7409027:
АЌ#
nyan_decoder_7409029:	Ќ
identity

identity_1ИҐ$nyan_decoder/StatefulPartitionedCallҐ$nyan_encoder/StatefulPartitionedCallу
$nyan_encoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2nyan_encoder_7408963nyan_encoder_7408965nyan_encoder_7408967nyan_encoder_7408969nyan_encoder_7408971nyan_encoder_7408973nyan_encoder_7408975nyan_encoder_7408977nyan_encoder_7408979nyan_encoder_7408981nyan_encoder_7408983nyan_encoder_7408985nyan_encoder_7408987nyan_encoder_7408989nyan_encoder_7408991nyan_encoder_7408993nyan_encoder_7408995nyan_encoder_7408997nyan_encoder_7408999nyan_encoder_7409001nyan_encoder_7409003nyan_encoder_7409005nyan_encoder_7409007nyan_encoder_7409009nyan_encoder_7409011nyan_encoder_7409013nyan_encoder_7409015*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408309О
$nyan_decoder/StatefulPartitionedCallStatefulPartitionedCall-nyan_encoder/StatefulPartitionedCall:output:0nyan_decoder_7409019nyan_decoder_7409021nyan_decoder_7409023nyan_decoder_7409025nyan_decoder_7409027nyan_decoder_7409029*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645}
IdentityIdentity-nyan_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фm

Identity_1Identity-nyan_encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ф
NoOpNoOp%^nyan_decoder/StatefulPartitionedCall%^nyan_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$nyan_decoder/StatefulPartitionedCall$nyan_decoder/StatefulPartitionedCall2L
$nyan_encoder/StatefulPartitionedCall$nyan_encoder/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs
ћ
Щ
)__inference_dense_2_layer_call_fn_7411640

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
„L
а
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7411492
inputs_0
inputs_1	
inputs_2

mask_0<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<=
ShapeShapeinputs_0*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  З
einsum/EinsumEinsummul:z:0inputs_0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd?
Shape_2Shapeinputs_0*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    j
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< \
mul_1MulBiasAdd:output:0mask_0*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_2/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€< 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
•E
Я
E__inference_sampling_layer_call_and_return_conditional_losses_7411785
inputs_0
inputs_1

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 

identity_2

identity_3ИҐAssignAddVariableOpҐAssignAddVariableOp_1ҐAssignAddVariableOp_2ҐAssignAddVariableOp_3ҐReadVariableOpҐdiv_no_nan/ReadVariableOpҐdiv_no_nan/ReadVariableOp_1Ґdiv_no_nan_1/ReadVariableOpҐdiv_no_nan_1/ReadVariableOp_1=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?µ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Јб<Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€@E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    O
ReadVariableOpReadVariableOpinputs*
_output_shapes
:*
dtype0L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<Y
mul_2Mulmul_2/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
:J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>H
subSub	mul_2:z:0sub/y:output:0*
T0*
_output_shapes
:P
MaximumMaximumConst_1:output:0sub:z:0*
T0*
_output_shapes
:R
MinimumMinimumConst:output:0Maximum:z:0*
T0*
_output_shapes
:L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?\
add_1AddV2add_1/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€@L
SquareSquareinputs_0*
T0*'
_output_shapes
:€€€€€€€€€@U
sub_1Sub	add_1:z:0
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@H
Exp_1Expinputs_1*
T0*'
_output_shapes
:€€€€€€€€€@T
sub_2Sub	sub_1:z:0	Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њ[
mul_3Mulmul_3/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :c
SumSum	mul_3:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: M
MeanMeanSum:output:0Const_2:output:0*
T0*
_output_shapes
: K
mul_4MulMinimum:z:0Mean:output:0*
T0*
_output_shapes
:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFT
truedivRealDiv	mul_4:z:0truediv/y:output:0*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: L
Sum_1SumMean:output:0range:output:0*
T0*
_output_shapes
: {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum_1:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: П
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0Ь
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0К
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: <
Rank_1Ranktruediv:z:0*
T0*
_output_shapes
: O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :v
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*#
_output_shapes
:€€€€€€€€€L
Sum_2Sumtruediv:z:0range_1:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*
_output_shapes
 *
dtype0<
Size_1Sizetruediv:z:0*
T0*
_output_shapes
: O
Cast_1CastSize_1:output:0*

DstT0*

SrcT0*
_output_shapes
: У
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_1:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0Ґ
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2^AssignAddVariableOp_3*
_output_shapes
: *
dtype0М
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0Е
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: X

Identity_2Identityadd:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@M

Identity_3Identitytruediv:z:0^NoOp*
T0*
_output_shapes
:≠
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€@:€€€€€€€€€@: : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32 
ReadVariableOpReadVariableOp26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_1:Q M
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/1:&"
 
_user_specified_nameinputs
ЩE
√

I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408507
x
a	
e
dense_7408439:
dense_7408441:#
ecc_conv_7408444:	А
ecc_conv_7408446:	А"
ecc_conv_7408448: 
ecc_conv_7408450: %
ecc_conv_1_7408453:	А!
ecc_conv_1_7408455:	А$
ecc_conv_1_7408457:   
ecc_conv_1_7408459: %
ecc_conv_2_7408462:	А!
ecc_conv_2_7408464:	А$
ecc_conv_2_7408466:   
ecc_conv_2_7408468: "
dense_1_7408472:	 А
dense_1_7408474:	А#
dense_2_7408478:
АА
dense_2_7408480:	А!
z_mean_7408483:	А@
z_mean_7408485:@$
z_log_var_7408488:	А@
z_log_var_7408490:@
sampling_7408493: 
sampling_7408495: 
sampling_7408497: 
sampling_7408499: 
sampling_7408501: 
identity

identity_1ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ ecc_conv/StatefulPartitionedCallҐ"ecc_conv_1/StatefulPartitionedCallҐ"ecc_conv_2/StatefulPartitionedCallҐ sampling/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall≈
graph_masking/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7407466r
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
graph_masking/strided_sliceStridedSlicex*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskО
dense/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0dense_7408439dense_7408441*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7407503с
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_7408444ecc_conv_7408446ecc_conv_7408448ecc_conv_7408450*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590А
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_1_7408453ecc_conv_1_7408455ecc_conv_1_7408457ecc_conv_1_7408459*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681В
"ecc_conv_2/StatefulPartitionedCallStatefulPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_2_7408462ecc_conv_2_7408464ecc_conv_2_7408466ecc_conv_2_7408468*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772п
global_sum_pool/PartitionedCallPartitionedCall+ecc_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788Х
dense_1/StatefulPartitionedCallStatefulPartitionedCall(global_sum_pool/PartitionedCall:output:0dense_1_7408472dense_1_7408474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801Ё
flatten/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7407813Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_7408478dense_2_7408480*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826Р
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_7408483z_mean_7408485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842Ь
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_7408488z_log_var_7408490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858€
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0sampling_7408493sampling_7408495sampling_7408497sampling_7408499sampling_7408501*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_7407942x
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@i

Identity_1Identity)sampling/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: €
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall#^ecc_conv_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall2H
"ecc_conv_2/StatefulPartitionedCall"ecc_conv_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:N J
+
_output_shapes
:€€€€€€€€€<

_user_specified_namex:NJ
+
_output_shapes
:€€€€€€€€€<<

_user_specified_namea:RN
/
_output_shapes
:€€€€€€€€€<<

_user_specified_namee
 	
х
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Э
џ
.__inference_nyan_encoder_layer_call_fn_7410395
inputs_0
inputs_1	
inputs_2
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 
identityИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7407957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
Љ
K
/__inference_graph_masking_layer_call_fn_7411237

inputs
identityЉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7407466d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
ћ
Щ
)__inference_dense_4_layer_call_fn_7411814

inputs
unknown:
АІ
	unknown_0:	І
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€І*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_7408620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€І`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
 	
х
C__inference_z_mean_layer_call_and_return_conditional_losses_7411670

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«	
—
*__inference_sampling_layer_call_fn_7411706
inputs_0
inputs_1

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputsunknown	unknown_0	unknown_1	unknown_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_7407942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€@:€€€€€€€€€@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
inputs/1:&"
 
_user_specified_nameinputs
у
ґ
@__inference_vae_layer_call_and_return_conditional_losses_7408810

inputs
inputs_1	
inputs_2&
nyan_encoder_7408739:"
nyan_encoder_7408741:'
nyan_encoder_7408743:	А#
nyan_encoder_7408745:	А&
nyan_encoder_7408747: "
nyan_encoder_7408749: '
nyan_encoder_7408751:	А#
nyan_encoder_7408753:	А&
nyan_encoder_7408755:  "
nyan_encoder_7408757: '
nyan_encoder_7408759:	А#
nyan_encoder_7408761:	А&
nyan_encoder_7408763:  "
nyan_encoder_7408765: '
nyan_encoder_7408767:	 А#
nyan_encoder_7408769:	А(
nyan_encoder_7408771:
АА#
nyan_encoder_7408773:	А'
nyan_encoder_7408775:	А@"
nyan_encoder_7408777:@'
nyan_encoder_7408779:	А@"
nyan_encoder_7408781:@
nyan_encoder_7408783: 
nyan_encoder_7408785: 
nyan_encoder_7408787: 
nyan_encoder_7408789: 
nyan_encoder_7408791: '
nyan_decoder_7408795:	@А#
nyan_decoder_7408797:	А(
nyan_decoder_7408799:
АІ#
nyan_decoder_7408801:	І(
nyan_decoder_7408803:
АЌ#
nyan_decoder_7408805:	Ќ
identity

identity_1ИҐ$nyan_decoder/StatefulPartitionedCallҐ$nyan_encoder/StatefulPartitionedCallу
$nyan_encoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2nyan_encoder_7408739nyan_encoder_7408741nyan_encoder_7408743nyan_encoder_7408745nyan_encoder_7408747nyan_encoder_7408749nyan_encoder_7408751nyan_encoder_7408753nyan_encoder_7408755nyan_encoder_7408757nyan_encoder_7408759nyan_encoder_7408761nyan_encoder_7408763nyan_encoder_7408765nyan_encoder_7408767nyan_encoder_7408769nyan_encoder_7408771nyan_encoder_7408773nyan_encoder_7408775nyan_encoder_7408777nyan_encoder_7408779nyan_encoder_7408781nyan_encoder_7408783nyan_encoder_7408785nyan_encoder_7408787nyan_encoder_7408789nyan_encoder_7408791*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7407957О
$nyan_decoder/StatefulPartitionedCallStatefulPartitionedCall-nyan_encoder/StatefulPartitionedCall:output:0nyan_decoder_7408795nyan_decoder_7408797nyan_decoder_7408799nyan_decoder_7408801nyan_decoder_7408803nyan_decoder_7408805*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645}
IdentityIdentity-nyan_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фm

Identity_1Identity-nyan_encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ф
NoOpNoOp%^nyan_decoder/StatefulPartitionedCall%^nyan_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$nyan_decoder/StatefulPartitionedCall$nyan_decoder/StatefulPartitionedCall2L
$nyan_encoder/StatefulPartitionedCall$nyan_encoder/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs
ё
∆
.__inference_nyan_encoder_layer_call_fn_7408015
x
a	
e
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallxaeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7407957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:€€€€€€€€€<

_user_specified_namex:NJ
+
_output_shapes
:€€€€€€€€€<<

_user_specified_namea:RN
/
_output_shapes
:€€€€€€€€€<<

_user_specified_namee
’

ч
D__inference_dense_1_layer_call_and_return_conditional_losses_7411620

inputs1
matmul_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Є
M
1__inference_global_sum_pool_layer_call_fn_7411594

inputs
identityЇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€< :S O
+
_output_shapes
:€€€€€€€€€< 
 
_user_specified_nameinputs
цШ
“!
@__inference_vae_layer_call_and_return_conditional_losses_7410333
inputs_0
inputs_1	
inputs_2F
4nyan_encoder_dense_tensordot_readvariableop_resource:@
2nyan_encoder_dense_biasadd_readvariableop_resource:R
?nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource:	АL
=nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource:	АG
5nyan_encoder_ecc_conv_shape_3_readvariableop_resource: C
5nyan_encoder_ecc_conv_biasadd_readvariableop_resource: T
Anyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	АN
?nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	АI
7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource:  E
7nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource: T
Anyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource:	АN
?nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource:	АI
7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource:  E
7nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource: F
3nyan_encoder_dense_1_matmul_readvariableop_resource:	 АC
4nyan_encoder_dense_1_biasadd_readvariableop_resource:	АG
3nyan_encoder_dense_2_matmul_readvariableop_resource:
ААC
4nyan_encoder_dense_2_biasadd_readvariableop_resource:	АE
2nyan_encoder_z_mean_matmul_readvariableop_resource:	А@A
3nyan_encoder_z_mean_biasadd_readvariableop_resource:@H
5nyan_encoder_z_log_var_matmul_readvariableop_resource:	А@D
6nyan_encoder_z_log_var_biasadd_readvariableop_resource:@7
-nyan_encoder_sampling_readvariableop_resource: <
2nyan_encoder_sampling_assignaddvariableop_resource: >
4nyan_encoder_sampling_assignaddvariableop_1_resource: >
4nyan_encoder_sampling_assignaddvariableop_2_resource: >
4nyan_encoder_sampling_assignaddvariableop_3_resource: F
3nyan_decoder_dense_3_matmul_readvariableop_resource:	@АC
4nyan_decoder_dense_3_biasadd_readvariableop_resource:	АG
3nyan_decoder_dense_4_matmul_readvariableop_resource:
АІC
4nyan_decoder_dense_4_biasadd_readvariableop_resource:	ІG
3nyan_decoder_dense_5_matmul_readvariableop_resource:
АЌC
4nyan_decoder_dense_5_biasadd_readvariableop_resource:	Ќ
identity

identity_1ИҐ+nyan_decoder/dense_3/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_3/MatMul/ReadVariableOpҐ+nyan_decoder/dense_4/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_4/MatMul/ReadVariableOpҐ+nyan_decoder/dense_5/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_5/MatMul/ReadVariableOpҐ)nyan_encoder/dense/BiasAdd/ReadVariableOpҐ+nyan_encoder/dense/Tensordot/ReadVariableOpҐ+nyan_encoder/dense_1/BiasAdd/ReadVariableOpҐ*nyan_encoder/dense_1/MatMul/ReadVariableOpҐ+nyan_encoder/dense_2/BiasAdd/ReadVariableOpҐ*nyan_encoder/dense_2/MatMul/ReadVariableOpҐ,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpҐ4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpҐ.nyan_encoder/ecc_conv/transpose/ReadVariableOpҐ.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpҐ8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpҐ0nyan_encoder/ecc_conv_1/transpose/ReadVariableOpҐ.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpҐ8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpҐ0nyan_encoder/ecc_conv_2/transpose/ReadVariableOpҐ)nyan_encoder/sampling/AssignAddVariableOpҐ+nyan_encoder/sampling/AssignAddVariableOp_1Ґ+nyan_encoder/sampling/AssignAddVariableOp_2Ґ+nyan_encoder/sampling/AssignAddVariableOp_3Ґ$nyan_encoder/sampling/ReadVariableOpҐ/nyan_encoder/sampling/div_no_nan/ReadVariableOpҐ1nyan_encoder/sampling/div_no_nan/ReadVariableOp_1Ґ1nyan_encoder/sampling/div_no_nan_1/ReadVariableOpҐ3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1Ґ-nyan_encoder/z_log_var/BiasAdd/ReadVariableOpҐ,nyan_encoder/z_log_var/MatMul/ReadVariableOpҐ*nyan_encoder/z_mean/BiasAdd/ReadVariableOpҐ)nyan_encoder/z_mean/MatMul/ReadVariableOp
.nyan_encoder/graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0nyan_encoder/graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€Б
0nyan_encoder/graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      џ
(nyan_encoder/graph_masking/strided_sliceStridedSliceinputs_07nyan_encoder/graph_masking/strided_slice/stack:output:09nyan_encoder/graph_masking/strided_slice/stack_1:output:09nyan_encoder/graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskБ
0nyan_encoder/graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Г
2nyan_encoder/graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2nyan_encoder/graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      б
*nyan_encoder/graph_masking/strided_slice_1StridedSliceinputs_09nyan_encoder/graph_masking/strided_slice_1/stack:output:0;nyan_encoder/graph_masking/strided_slice_1/stack_1:output:0;nyan_encoder/graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_mask†
+nyan_encoder/dense/Tensordot/ReadVariableOpReadVariableOp4nyan_encoder_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!nyan_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!nyan_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Г
"nyan_encoder/dense/Tensordot/ShapeShape1nyan_encoder/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:l
*nyan_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%nyan_encoder/dense/Tensordot/GatherV2GatherV2+nyan_encoder/dense/Tensordot/Shape:output:0*nyan_encoder/dense/Tensordot/free:output:03nyan_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,nyan_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'nyan_encoder/dense/Tensordot/GatherV2_1GatherV2+nyan_encoder/dense/Tensordot/Shape:output:0*nyan_encoder/dense/Tensordot/axes:output:05nyan_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"nyan_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!nyan_encoder/dense/Tensordot/ProdProd.nyan_encoder/dense/Tensordot/GatherV2:output:0+nyan_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$nyan_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#nyan_encoder/dense/Tensordot/Prod_1Prod0nyan_encoder/dense/Tensordot/GatherV2_1:output:0-nyan_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(nyan_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#nyan_encoder/dense/Tensordot/concatConcatV2*nyan_encoder/dense/Tensordot/free:output:0*nyan_encoder/dense/Tensordot/axes:output:01nyan_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"nyan_encoder/dense/Tensordot/stackPack*nyan_encoder/dense/Tensordot/Prod:output:0,nyan_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
&nyan_encoder/dense/Tensordot/transpose	Transpose1nyan_encoder/graph_masking/strided_slice:output:0,nyan_encoder/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<√
$nyan_encoder/dense/Tensordot/ReshapeReshape*nyan_encoder/dense/Tensordot/transpose:y:0+nyan_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€√
#nyan_encoder/dense/Tensordot/MatMulMatMul-nyan_encoder/dense/Tensordot/Reshape:output:03nyan_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€n
$nyan_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*nyan_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%nyan_encoder/dense/Tensordot/concat_1ConcatV2.nyan_encoder/dense/Tensordot/GatherV2:output:0-nyan_encoder/dense/Tensordot/Const_2:output:03nyan_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Љ
nyan_encoder/dense/TensordotReshape-nyan_encoder/dense/Tensordot/MatMul:product:0.nyan_encoder/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<Ш
)nyan_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2nyan_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
nyan_encoder/dense/BiasAddBiasAdd%nyan_encoder/dense/Tensordot:output:01nyan_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<Ч
(nyan_encoder/dense/leaky_re_lu/LeakyRelu	LeakyRelu#nyan_encoder/dense/BiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=q
nyan_encoder/ecc_conv/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<Б
nyan_encoder/ecc_conv/ShapeShape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:|
)nyan_encoder/ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€~
+nyan_encoder/ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€u
+nyan_encoder/ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#nyan_encoder/ecc_conv/strided_sliceStridedSlice$nyan_encoder/ecc_conv/Shape:output:02nyan_encoder/ecc_conv/strided_slice/stack:output:04nyan_encoder/ecc_conv/strided_slice/stack_1:output:04nyan_encoder/ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskГ
nyan_encoder/ecc_conv/Shape_1Shape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-nyan_encoder/ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv/strided_slice_1StridedSlice&nyan_encoder/ecc_conv/Shape_1:output:04nyan_encoder/ecc_conv/strided_slice_1/stack:output:06nyan_encoder/ecc_conv/strided_slice_1/stack_1:output:06nyan_encoder/ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0v
,nyan_encoder/ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Б
,nyan_encoder/ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          e
-nyan_encoder/ecc_conv/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:w
5nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ≥
0nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2GatherV26nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
2nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV26nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0@nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-nyan_encoder/ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: »
,nyan_encoder/ecc_conv/FGN_out/Tensordot/ProdProd9nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:06nyan_encoder/ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1Prod;nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1:output:08nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.nyan_encoder/ecc_conv/FGN_out/Tensordot/concatConcatV25nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0<nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:”
-nyan_encoder/ecc_conv/FGN_out/Tensordot/stackPack5nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod:output:07nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ї
1nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose	Transposeinputs_27nyan_encoder/ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<д
/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReshapeReshape5nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose:y:06nyan_encoder/ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€е
.nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMulMatMul8nyan_encoder/ecc_conv/FGN_out/Tensordot/Reshape:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аz
/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аw
5nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1ConcatV29nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:в
'nyan_encoder/ecc_conv/FGN_out/TensordotReshape8nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMul:product:09nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аѓ
4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp=nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0џ
%nyan_encoder/ecc_conv/FGN_out/BiasAddBiasAdd0nyan_encoder/ecc_conv/FGN_out/Tensordot:output:0<nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аp
%nyan_encoder/ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€g
%nyan_encoder/ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ≈
#nyan_encoder/ecc_conv/Reshape/shapePack.nyan_encoder/ecc_conv/Reshape/shape/0:output:0,nyan_encoder/ecc_conv/strided_slice:output:0,nyan_encoder/ecc_conv/strided_slice:output:0.nyan_encoder/ecc_conv/Reshape/shape/3:output:0.nyan_encoder/ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ƒ
nyan_encoder/ecc_conv/ReshapeReshape.nyan_encoder/ecc_conv/FGN_out/BiasAdd:output:0,nyan_encoder/ecc_conv/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< А
+nyan_encoder/ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            В
-nyan_encoder/ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            В
-nyan_encoder/ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         р
%nyan_encoder/ecc_conv/strided_slice_2StridedSlicenyan_encoder/ecc_conv/Cast:y:04nyan_encoder/ecc_conv/strided_slice_2/stack:output:06nyan_encoder/ecc_conv/strided_slice_2/stack_1:output:06nyan_encoder/ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskґ
nyan_encoder/ecc_conv/mulMul&nyan_encoder/ecc_conv/Reshape:output:0.nyan_encoder/ecc_conv/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< б
#nyan_encoder/ecc_conv/einsum/EinsumEinsumnyan_encoder/ecc_conv/mul:z:06nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdГ
nyan_encoder/ecc_conv/Shape_2Shape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:
nyan_encoder/ecc_conv/unstackUnpack&nyan_encoder/ecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numҐ
,nyan_encoder/ecc_conv/Shape_3/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0n
nyan_encoder/ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       
nyan_encoder/ecc_conv/unstack_1Unpack&nyan_encoder/ecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
numv
%nyan_encoder/ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ƒ
nyan_encoder/ecc_conv/Reshape_1Reshape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0.nyan_encoder/ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
.nyan_encoder/ecc_conv/transpose/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0u
$nyan_encoder/ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Љ
nyan_encoder/ecc_conv/transpose	Transpose6nyan_encoder/ecc_conv/transpose/ReadVariableOp:value:0-nyan_encoder/ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: v
%nyan_encoder/ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€®
nyan_encoder/ecc_conv/Reshape_2Reshape#nyan_encoder/ecc_conv/transpose:y:0.nyan_encoder/ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: ђ
nyan_encoder/ecc_conv/MatMulMatMul(nyan_encoder/ecc_conv/Reshape_1:output:0(nyan_encoder/ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ i
'nyan_encoder/ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<i
'nyan_encoder/ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : з
%nyan_encoder/ecc_conv/Reshape_3/shapePack&nyan_encoder/ecc_conv/unstack:output:00nyan_encoder/ecc_conv/Reshape_3/shape/1:output:00nyan_encoder/ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Є
nyan_encoder/ecc_conv/Reshape_3Reshape&nyan_encoder/ecc_conv/MatMul:product:0.nyan_encoder/ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ∞
nyan_encoder/ecc_conv/addAddV2,nyan_encoder/ecc_conv/einsum/Einsum:output:0(nyan_encoder/ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ю
,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≥
nyan_encoder/ecc_conv/BiasAddBiasAddnyan_encoder/ecc_conv/add:z:04nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< µ
nyan_encoder/ecc_conv/mul_1Mul&nyan_encoder/ecc_conv/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ш
-nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu	LeakyRelunyan_encoder/ecc_conv/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=s
nyan_encoder/ecc_conv_1/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<И
nyan_encoder/ecc_conv_1/ShapeShape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€А
-nyan_encoder/ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv_1/strided_sliceStridedSlice&nyan_encoder/ecc_conv_1/Shape:output:04nyan_encoder/ecc_conv_1/strided_slice/stack:output:06nyan_encoder/ecc_conv_1/strided_slice/stack_1:output:06nyan_encoder/ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
nyan_encoder/ecc_conv_1/Shape_1Shape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:А
-nyan_encoder/ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€y
/nyan_encoder/ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/nyan_encoder/ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'nyan_encoder/ecc_conv_1/strided_slice_1StridedSlice(nyan_encoder/ecc_conv_1/Shape_1:output:06nyan_encoder/ecc_conv_1/strided_slice_1/stack:output:08nyan_encoder/ecc_conv_1/strided_slice_1/stack_1:output:08nyan_encoder/ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOpAnyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0x
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Г
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          g
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:y
7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
2nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV28nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
4nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV28nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0Bnyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ProdProd;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ‘
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1Prod=nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concatConcatV27nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0>nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stackPack7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod:output:09nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:њ
3nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose	Transposeinputs_29nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<к
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReshapeReshape7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose:y:08nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€л
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMulMatMul:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Reshape:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А|
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аy
7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
2nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
)nyan_encoder/ecc_conv_1/FGN_out/TensordotReshape:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMul:product:0;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<А≥
6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
'nyan_encoder/ecc_conv_1/FGN_out/BiasAddBiasAdd2nyan_encoder/ecc_conv_1/FGN_out/Tensordot:output:0>nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аr
'nyan_encoder/ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€i
'nyan_encoder/ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : —
%nyan_encoder/ecc_conv_1/Reshape/shapePack0nyan_encoder/ecc_conv_1/Reshape/shape/0:output:0.nyan_encoder/ecc_conv_1/strided_slice:output:0.nyan_encoder/ecc_conv_1/strided_slice:output:00nyan_encoder/ecc_conv_1/Reshape/shape/3:output:00nyan_encoder/ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
: 
nyan_encoder/ecc_conv_1/ReshapeReshape0nyan_encoder/ecc_conv_1/FGN_out/BiasAdd:output:0.nyan_encoder/ecc_conv_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  В
-nyan_encoder/ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ъ
'nyan_encoder/ecc_conv_1/strided_slice_2StridedSlice nyan_encoder/ecc_conv_1/Cast:y:06nyan_encoder/ecc_conv_1/strided_slice_2/stack:output:08nyan_encoder/ecc_conv_1/strided_slice_2/stack_1:output:08nyan_encoder/ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskЉ
nyan_encoder/ecc_conv_1/mulMul(nyan_encoder/ecc_conv_1/Reshape:output:00nyan_encoder/ecc_conv_1/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  к
%nyan_encoder/ecc_conv_1/einsum/EinsumEinsumnyan_encoder/ecc_conv_1/mul:z:0;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdК
nyan_encoder/ecc_conv_1/Shape_2Shape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:Г
nyan_encoder/ecc_conv_1/unstackUnpack(nyan_encoder/ecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num¶
.nyan_encoder/ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0p
nyan_encoder/ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Г
!nyan_encoder/ecc_conv_1/unstack_1Unpack(nyan_encoder/ecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numx
'nyan_encoder/ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ќ
!nyan_encoder/ecc_conv_1/Reshape_1Reshape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:00nyan_encoder/ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ®
0nyan_encoder/ecc_conv_1/transpose/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0w
&nyan_encoder/ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!nyan_encoder/ecc_conv_1/transpose	Transpose8nyan_encoder/ecc_conv_1/transpose/ReadVariableOp:value:0/nyan_encoder/ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  x
'nyan_encoder/ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ѓ
!nyan_encoder/ecc_conv_1/Reshape_2Reshape%nyan_encoder/ecc_conv_1/transpose:y:00nyan_encoder/ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  ≤
nyan_encoder/ecc_conv_1/MatMulMatMul*nyan_encoder/ecc_conv_1/Reshape_1:output:0*nyan_encoder/ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
)nyan_encoder/ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<k
)nyan_encoder/ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
'nyan_encoder/ecc_conv_1/Reshape_3/shapePack(nyan_encoder/ecc_conv_1/unstack:output:02nyan_encoder/ecc_conv_1/Reshape_3/shape/1:output:02nyan_encoder/ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Њ
!nyan_encoder/ecc_conv_1/Reshape_3Reshape(nyan_encoder/ecc_conv_1/MatMul:product:00nyan_encoder/ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ґ
nyan_encoder/ecc_conv_1/addAddV2.nyan_encoder/ecc_conv_1/einsum/Einsum:output:0*nyan_encoder/ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ґ
.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
nyan_encoder/ecc_conv_1/BiasAddBiasAddnyan_encoder/ecc_conv_1/add:z:06nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< є
nyan_encoder/ecc_conv_1/mul_1Mul(nyan_encoder/ecc_conv_1/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ь
/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu	LeakyRelu!nyan_encoder/ecc_conv_1/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=s
nyan_encoder/ecc_conv_2/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<К
nyan_encoder/ecc_conv_2/ShapeShape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€А
-nyan_encoder/ecc_conv_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv_2/strided_sliceStridedSlice&nyan_encoder/ecc_conv_2/Shape:output:04nyan_encoder/ecc_conv_2/strided_slice/stack:output:06nyan_encoder/ecc_conv_2/strided_slice/stack_1:output:06nyan_encoder/ecc_conv_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
nyan_encoder/ecc_conv_2/Shape_1Shape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:А
-nyan_encoder/ecc_conv_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€y
/nyan_encoder/ecc_conv_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/nyan_encoder/ecc_conv_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'nyan_encoder/ecc_conv_2/strided_slice_1StridedSlice(nyan_encoder/ecc_conv_2/Shape_1:output:06nyan_encoder/ecc_conv_2/strided_slice_1/stack:output:08nyan_encoder/ecc_conv_2/strided_slice_1/stack_1:output:08nyan_encoder/ecc_conv_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpReadVariableOpAnyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0x
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Г
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          g
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:y
7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
2nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2GatherV28nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
4nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1GatherV28nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0Bnyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ProdProd;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ‘
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1Prod=nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1:output:0:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concatConcatV27nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0>nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stackPack7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod:output:09nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:њ
3nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose	Transposeinputs_29nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<к
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReshapeReshape7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose:y:08nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€л
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMulMatMul:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Reshape:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А|
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аy
7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
2nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1ConcatV2;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
)nyan_encoder/ecc_conv_2/FGN_out/TensordotReshape:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMul:product:0;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<А≥
6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
'nyan_encoder/ecc_conv_2/FGN_out/BiasAddBiasAdd2nyan_encoder/ecc_conv_2/FGN_out/Tensordot:output:0>nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аr
'nyan_encoder/ecc_conv_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€i
'nyan_encoder/ecc_conv_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : —
%nyan_encoder/ecc_conv_2/Reshape/shapePack0nyan_encoder/ecc_conv_2/Reshape/shape/0:output:0.nyan_encoder/ecc_conv_2/strided_slice:output:0.nyan_encoder/ecc_conv_2/strided_slice:output:00nyan_encoder/ecc_conv_2/Reshape/shape/3:output:00nyan_encoder/ecc_conv_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
: 
nyan_encoder/ecc_conv_2/ReshapeReshape0nyan_encoder/ecc_conv_2/FGN_out/BiasAdd:output:0.nyan_encoder/ecc_conv_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  В
-nyan_encoder/ecc_conv_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ъ
'nyan_encoder/ecc_conv_2/strided_slice_2StridedSlice nyan_encoder/ecc_conv_2/Cast:y:06nyan_encoder/ecc_conv_2/strided_slice_2/stack:output:08nyan_encoder/ecc_conv_2/strided_slice_2/stack_1:output:08nyan_encoder/ecc_conv_2/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskЉ
nyan_encoder/ecc_conv_2/mulMul(nyan_encoder/ecc_conv_2/Reshape:output:00nyan_encoder/ecc_conv_2/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  м
%nyan_encoder/ecc_conv_2/einsum/EinsumEinsumnyan_encoder/ecc_conv_2/mul:z:0=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdМ
nyan_encoder/ecc_conv_2/Shape_2Shape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:Г
nyan_encoder/ecc_conv_2/unstackUnpack(nyan_encoder/ecc_conv_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num¶
.nyan_encoder/ecc_conv_2/Shape_3/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0p
nyan_encoder/ecc_conv_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Г
!nyan_encoder/ecc_conv_2/unstack_1Unpack(nyan_encoder/ecc_conv_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
numx
'nyan_encoder/ecc_conv_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ѕ
!nyan_encoder/ecc_conv_2/Reshape_1Reshape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:00nyan_encoder/ecc_conv_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ®
0nyan_encoder/ecc_conv_2/transpose/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0w
&nyan_encoder/ecc_conv_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!nyan_encoder/ecc_conv_2/transpose	Transpose8nyan_encoder/ecc_conv_2/transpose/ReadVariableOp:value:0/nyan_encoder/ecc_conv_2/transpose/perm:output:0*
T0*
_output_shapes

:  x
'nyan_encoder/ecc_conv_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ѓ
!nyan_encoder/ecc_conv_2/Reshape_2Reshape%nyan_encoder/ecc_conv_2/transpose:y:00nyan_encoder/ecc_conv_2/Reshape_2/shape:output:0*
T0*
_output_shapes

:  ≤
nyan_encoder/ecc_conv_2/MatMulMatMul*nyan_encoder/ecc_conv_2/Reshape_1:output:0*nyan_encoder/ecc_conv_2/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
)nyan_encoder/ecc_conv_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<k
)nyan_encoder/ecc_conv_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
'nyan_encoder/ecc_conv_2/Reshape_3/shapePack(nyan_encoder/ecc_conv_2/unstack:output:02nyan_encoder/ecc_conv_2/Reshape_3/shape/1:output:02nyan_encoder/ecc_conv_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Њ
!nyan_encoder/ecc_conv_2/Reshape_3Reshape(nyan_encoder/ecc_conv_2/MatMul:product:00nyan_encoder/ecc_conv_2/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ґ
nyan_encoder/ecc_conv_2/addAddV2.nyan_encoder/ecc_conv_2/einsum/Einsum:output:0*nyan_encoder/ecc_conv_2/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ґ
.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
nyan_encoder/ecc_conv_2/BiasAddBiasAddnyan_encoder/ecc_conv_2/add:z:06nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< є
nyan_encoder/ecc_conv_2/mul_1Mul(nyan_encoder/ecc_conv_2/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ь
/nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu	LeakyRelu!nyan_encoder/ecc_conv_2/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=}
2nyan_encoder/global_sum_pool/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€’
 nyan_encoder/global_sum_pool/SumSum=nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu:activations:0;nyan_encoder/global_sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Я
*nyan_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3nyan_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0Ј
nyan_encoder/dense_1/MatMulMatMul)nyan_encoder/global_sum_pool/Sum:output:02nyan_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4nyan_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_encoder/dense_1/BiasAddBiasAdd%nyan_encoder/dense_1/MatMul:product:03nyan_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu	LeakyRelu%nyan_encoder/dense_1/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=k
nyan_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ї
nyan_encoder/flatten/ReshapeReshape:nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu:activations:0#nyan_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А†
*nyan_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3nyan_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0≥
nyan_encoder/dense_2/MatMulMatMul%nyan_encoder/flatten/Reshape:output:02nyan_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4nyan_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_encoder/dense_2/BiasAddBiasAdd%nyan_encoder/dense_2/MatMul:product:03nyan_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu	LeakyRelu%nyan_encoder/dense_2/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=Э
)nyan_encoder/z_mean/MatMul/ReadVariableOpReadVariableOp2nyan_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0≈
nyan_encoder/z_mean/MatMulMatMul:nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:01nyan_encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
*nyan_encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp3nyan_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0≤
nyan_encoder/z_mean/BiasAddBiasAdd$nyan_encoder/z_mean/MatMul:product:02nyan_encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@£
,nyan_encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp5nyan_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Ћ
nyan_encoder/z_log_var/MatMulMatMul:nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:04nyan_encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@†
-nyan_encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp6nyan_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ї
nyan_encoder/z_log_var/BiasAddBiasAdd'nyan_encoder/z_log_var/MatMul:product:05nyan_encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@o
nyan_encoder/sampling/ShapeShape$nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:s
)nyan_encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nyan_encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nyan_encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#nyan_encoder/sampling/strided_sliceStridedSlice$nyan_encoder/sampling/Shape:output:02nyan_encoder/sampling/strided_slice/stack:output:04nyan_encoder/sampling/strided_slice/stack_1:output:04nyan_encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
nyan_encoder/sampling/Shape_1Shape$nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:u
+nyan_encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-nyan_encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-nyan_encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/sampling/strided_slice_1StridedSlice&nyan_encoder/sampling/Shape_1:output:04nyan_encoder/sampling/strided_slice_1/stack:output:06nyan_encoder/sampling/strided_slice_1/stack_1:output:06nyan_encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
)nyan_encoder/sampling/random_normal/shapePack,nyan_encoder/sampling/strided_slice:output:0.nyan_encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:m
(nyan_encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    o
*nyan_encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?в
8nyan_encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal2nyan_encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ь±КЎ
'nyan_encoder/sampling/random_normal/mulMulAnyan_encoder/sampling/random_normal/RandomStandardNormal:output:03nyan_encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
#nyan_encoder/sampling/random_normalAddV2+nyan_encoder/sampling/random_normal/mul:z:01nyan_encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@`
nyan_encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
nyan_encoder/sampling/mulMul$nyan_encoder/sampling/mul/x:output:0'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
nyan_encoder/sampling/ExpExpnyan_encoder/sampling/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
nyan_encoder/sampling/mul_1Mulnyan_encoder/sampling/Exp:y:0'nyan_encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ы
nyan_encoder/sampling/addAddV2$nyan_encoder/z_mean/BiasAdd:output:0nyan_encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@`
nyan_encoder/sampling/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
nyan_encoder/sampling/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    К
$nyan_encoder/sampling/ReadVariableOpReadVariableOp-nyan_encoder_sampling_readvariableop_resource*
_output_shapes
: *
dtype0b
nyan_encoder/sampling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<Щ
nyan_encoder/sampling/mul_2Mul&nyan_encoder/sampling/mul_2/x:output:0,nyan_encoder/sampling/ReadVariableOp:value:0*
T0*
_output_shapes
: `
nyan_encoder/sampling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>И
nyan_encoder/sampling/subSubnyan_encoder/sampling/mul_2:z:0$nyan_encoder/sampling/sub/y:output:0*
T0*
_output_shapes
: Р
nyan_encoder/sampling/MaximumMaximum&nyan_encoder/sampling/Const_1:output:0nyan_encoder/sampling/sub:z:0*
T0*
_output_shapes
: Т
nyan_encoder/sampling/MinimumMinimum$nyan_encoder/sampling/Const:output:0!nyan_encoder/sampling/Maximum:z:0*
T0*
_output_shapes
: b
nyan_encoder/sampling/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
nyan_encoder/sampling/add_1AddV2&nyan_encoder/sampling/add_1/x:output:0'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
nyan_encoder/sampling/SquareSquare$nyan_encoder/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
nyan_encoder/sampling/sub_1Subnyan_encoder/sampling/add_1:z:0 nyan_encoder/sampling/Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@}
nyan_encoder/sampling/Exp_1Exp'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
nyan_encoder/sampling/sub_2Subnyan_encoder/sampling/sub_1:z:0nyan_encoder/sampling/Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@b
nyan_encoder/sampling/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њЭ
nyan_encoder/sampling/mul_3Mul&nyan_encoder/sampling/mul_3/x:output:0nyan_encoder/sampling/sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@m
+nyan_encoder/sampling/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :•
nyan_encoder/sampling/SumSumnyan_encoder/sampling/mul_3:z:04nyan_encoder/sampling/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
nyan_encoder/sampling/Const_2Const*
_output_shapes
:*
dtype0*
valueB: П
nyan_encoder/sampling/MeanMean"nyan_encoder/sampling/Sum:output:0&nyan_encoder/sampling/Const_2:output:0*
T0*
_output_shapes
: Л
nyan_encoder/sampling/mul_4Mul!nyan_encoder/sampling/Minimum:z:0#nyan_encoder/sampling/Mean:output:0*
T0*
_output_shapes
: d
nyan_encoder/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFФ
nyan_encoder/sampling/truedivRealDivnyan_encoder/sampling/mul_4:z:0(nyan_encoder/sampling/truediv/y:output:0*
T0*
_output_shapes
: \
nyan_encoder/sampling/RankConst*
_output_shapes
: *
dtype0*
value	B : c
!nyan_encoder/sampling/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!nyan_encoder/sampling/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ї
nyan_encoder/sampling/rangeRange*nyan_encoder/sampling/range/start:output:0#nyan_encoder/sampling/Rank:output:0*nyan_encoder/sampling/range/delta:output:0*
_output_shapes
: О
nyan_encoder/sampling/Sum_1Sum#nyan_encoder/sampling/Mean:output:0$nyan_encoder/sampling/range:output:0*
T0*
_output_shapes
: љ
)nyan_encoder/sampling/AssignAddVariableOpAssignAddVariableOp2nyan_encoder_sampling_assignaddvariableop_resource$nyan_encoder/sampling/Sum_1:output:0*
_output_shapes
 *
dtype0\
nyan_encoder/sampling/SizeConst*
_output_shapes
: *
dtype0*
value	B :w
nyan_encoder/sampling/CastCast#nyan_encoder/sampling/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: з
+nyan_encoder/sampling/AssignAddVariableOp_1AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_1_resourcenyan_encoder/sampling/Cast:y:0*^nyan_encoder/sampling/AssignAddVariableOp*
_output_shapes
 *
dtype0ф
/nyan_encoder/sampling/div_no_nan/ReadVariableOpReadVariableOp2nyan_encoder_sampling_assignaddvariableop_resource*^nyan_encoder/sampling/AssignAddVariableOp,^nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0ћ
1nyan_encoder/sampling/div_no_nan/ReadVariableOp_1ReadVariableOp4nyan_encoder_sampling_assignaddvariableop_1_resource,^nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ѕ
 nyan_encoder/sampling/div_no_nanDivNoNan7nyan_encoder/sampling/div_no_nan/ReadVariableOp:value:09nyan_encoder/sampling/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: q
nyan_encoder/sampling/IdentityIdentity$nyan_encoder/sampling/div_no_nan:z:0*
T0*
_output_shapes
: ^
nyan_encoder/sampling/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : e
#nyan_encoder/sampling/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : e
#nyan_encoder/sampling/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :√
nyan_encoder/sampling/range_1Range,nyan_encoder/sampling/range_1/start:output:0%nyan_encoder/sampling/Rank_1:output:0,nyan_encoder/sampling/range_1/delta:output:0*
_output_shapes
: О
nyan_encoder/sampling/Sum_2Sum!nyan_encoder/sampling/truediv:z:0&nyan_encoder/sampling/range_1:output:0*
T0*
_output_shapes
: Ѕ
+nyan_encoder/sampling/AssignAddVariableOp_2AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_2_resource$nyan_encoder/sampling/Sum_2:output:0*
_output_shapes
 *
dtype0^
nyan_encoder/sampling/Size_1Const*
_output_shapes
: *
dtype0*
value	B :{
nyan_encoder/sampling/Cast_1Cast%nyan_encoder/sampling/Size_1:output:0*

DstT0*

SrcT0*
_output_shapes
: л
+nyan_encoder/sampling/AssignAddVariableOp_3AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_3_resource nyan_encoder/sampling/Cast_1:y:0,^nyan_encoder/sampling/AssignAddVariableOp_2*
_output_shapes
 *
dtype0ъ
1nyan_encoder/sampling/div_no_nan_1/ReadVariableOpReadVariableOp4nyan_encoder_sampling_assignaddvariableop_2_resource,^nyan_encoder/sampling/AssignAddVariableOp_2,^nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0ќ
3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1ReadVariableOp4nyan_encoder_sampling_assignaddvariableop_3_resource,^nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0«
"nyan_encoder/sampling/div_no_nan_1DivNoNan9nyan_encoder/sampling/div_no_nan_1/ReadVariableOp:value:0;nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: u
 nyan_encoder/sampling/Identity_1Identity&nyan_encoder/sampling/div_no_nan_1:z:0*
T0*
_output_shapes
: Я
*nyan_decoder/dense_3/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Ђ
nyan_decoder/dense_3/MatMulMatMulnyan_encoder/sampling/add:z:02nyan_decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_decoder/dense_3/BiasAddBiasAdd%nyan_decoder/dense_3/MatMul:product:03nyan_decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu	LeakyRelu%nyan_decoder/dense_3/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=†
*nyan_decoder/dense_4/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0»
nyan_decoder/dense_4/MatMulMatMul:nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:02nyan_decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІЭ
+nyan_decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0ґ
nyan_decoder/dense_4/BiasAddBiasAdd%nyan_decoder/dense_4/MatMul:product:03nyan_decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІБ
nyan_decoder/dense_4/SigmoidSigmoid%nyan_decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€І†
*nyan_decoder/dense_5/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0»
nyan_decoder/dense_5/MatMulMatMul:nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:02nyan_decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЌЭ
+nyan_decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0ґ
nyan_decoder/dense_5/BiasAddBiasAdd%nyan_decoder/dense_5/MatMul:product:03nyan_decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќc
nyan_decoder/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€«
nyan_decoder/concatConcatV2 nyan_decoder/dense_4/Sigmoid:y:0%nyan_decoder/dense_5/BiasAdd:output:0!nyan_decoder/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€фl
IdentityIdentitynyan_decoder/concat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фa

Identity_1Identity!nyan_encoder/sampling/truediv:z:0^NoOp*
T0*
_output_shapes
: ћ
NoOpNoOp,^nyan_decoder/dense_3/BiasAdd/ReadVariableOp+^nyan_decoder/dense_3/MatMul/ReadVariableOp,^nyan_decoder/dense_4/BiasAdd/ReadVariableOp+^nyan_decoder/dense_4/MatMul/ReadVariableOp,^nyan_decoder/dense_5/BiasAdd/ReadVariableOp+^nyan_decoder/dense_5/MatMul/ReadVariableOp*^nyan_encoder/dense/BiasAdd/ReadVariableOp,^nyan_encoder/dense/Tensordot/ReadVariableOp,^nyan_encoder/dense_1/BiasAdd/ReadVariableOp+^nyan_encoder/dense_1/MatMul/ReadVariableOp,^nyan_encoder/dense_2/BiasAdd/ReadVariableOp+^nyan_encoder/dense_2/MatMul/ReadVariableOp-^nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp5^nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp/^nyan_encoder/ecc_conv/transpose/ReadVariableOp/^nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp9^nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp1^nyan_encoder/ecc_conv_1/transpose/ReadVariableOp/^nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp9^nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp1^nyan_encoder/ecc_conv_2/transpose/ReadVariableOp*^nyan_encoder/sampling/AssignAddVariableOp,^nyan_encoder/sampling/AssignAddVariableOp_1,^nyan_encoder/sampling/AssignAddVariableOp_2,^nyan_encoder/sampling/AssignAddVariableOp_3%^nyan_encoder/sampling/ReadVariableOp0^nyan_encoder/sampling/div_no_nan/ReadVariableOp2^nyan_encoder/sampling/div_no_nan/ReadVariableOp_12^nyan_encoder/sampling/div_no_nan_1/ReadVariableOp4^nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1.^nyan_encoder/z_log_var/BiasAdd/ReadVariableOp-^nyan_encoder/z_log_var/MatMul/ReadVariableOp+^nyan_encoder/z_mean/BiasAdd/ReadVariableOp*^nyan_encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+nyan_decoder/dense_3/BiasAdd/ReadVariableOp+nyan_decoder/dense_3/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_3/MatMul/ReadVariableOp*nyan_decoder/dense_3/MatMul/ReadVariableOp2Z
+nyan_decoder/dense_4/BiasAdd/ReadVariableOp+nyan_decoder/dense_4/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_4/MatMul/ReadVariableOp*nyan_decoder/dense_4/MatMul/ReadVariableOp2Z
+nyan_decoder/dense_5/BiasAdd/ReadVariableOp+nyan_decoder/dense_5/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_5/MatMul/ReadVariableOp*nyan_decoder/dense_5/MatMul/ReadVariableOp2V
)nyan_encoder/dense/BiasAdd/ReadVariableOp)nyan_encoder/dense/BiasAdd/ReadVariableOp2Z
+nyan_encoder/dense/Tensordot/ReadVariableOp+nyan_encoder/dense/Tensordot/ReadVariableOp2Z
+nyan_encoder/dense_1/BiasAdd/ReadVariableOp+nyan_encoder/dense_1/BiasAdd/ReadVariableOp2X
*nyan_encoder/dense_1/MatMul/ReadVariableOp*nyan_encoder/dense_1/MatMul/ReadVariableOp2Z
+nyan_encoder/dense_2/BiasAdd/ReadVariableOp+nyan_encoder/dense_2/BiasAdd/ReadVariableOp2X
*nyan_encoder/dense_2/MatMul/ReadVariableOp*nyan_encoder/dense_2/MatMul/ReadVariableOp2\
,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp2l
4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp2`
.nyan_encoder/ecc_conv/transpose/ReadVariableOp.nyan_encoder/ecc_conv/transpose/ReadVariableOp2`
.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2t
8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2d
0nyan_encoder/ecc_conv_1/transpose/ReadVariableOp0nyan_encoder/ecc_conv_1/transpose/ReadVariableOp2`
.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp2t
8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp2d
0nyan_encoder/ecc_conv_2/transpose/ReadVariableOp0nyan_encoder/ecc_conv_2/transpose/ReadVariableOp2V
)nyan_encoder/sampling/AssignAddVariableOp)nyan_encoder/sampling/AssignAddVariableOp2Z
+nyan_encoder/sampling/AssignAddVariableOp_1+nyan_encoder/sampling/AssignAddVariableOp_12Z
+nyan_encoder/sampling/AssignAddVariableOp_2+nyan_encoder/sampling/AssignAddVariableOp_22Z
+nyan_encoder/sampling/AssignAddVariableOp_3+nyan_encoder/sampling/AssignAddVariableOp_32L
$nyan_encoder/sampling/ReadVariableOp$nyan_encoder/sampling/ReadVariableOp2b
/nyan_encoder/sampling/div_no_nan/ReadVariableOp/nyan_encoder/sampling/div_no_nan/ReadVariableOp2f
1nyan_encoder/sampling/div_no_nan/ReadVariableOp_11nyan_encoder/sampling/div_no_nan/ReadVariableOp_12f
1nyan_encoder/sampling/div_no_nan_1/ReadVariableOp1nyan_encoder/sampling/div_no_nan_1/ReadVariableOp2j
3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_13nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_12^
-nyan_encoder/z_log_var/BiasAdd/ReadVariableOp-nyan_encoder/z_log_var/BiasAdd/ReadVariableOp2\
,nyan_encoder/z_log_var/MatMul/ReadVariableOp,nyan_encoder/z_log_var/MatMul/ReadVariableOp2X
*nyan_encoder/z_mean/BiasAdd/ReadVariableOp*nyan_encoder/z_mean/BiasAdd/ReadVariableOp2V
)nyan_encoder/z_mean/MatMul/ReadVariableOp)nyan_encoder/z_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
’

ч
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801

inputs1
matmul_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ќ	
ш
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
’L
ё
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7411395
inputs_0
inputs_1	
inputs_2

mask_0<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<=
ShapeShapeinputs_0*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< З
einsum/EinsumEinsummul:z:0inputs_0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd?
Shape_2Shapeinputs_0*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   j
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

: j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< \
mul_1MulBiasAdd:output:0mask_0*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_1/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
ћ
Щ
)__inference_dense_5_layer_call_fn_7411834

inputs
unknown:
АЌ
	unknown_0:	Ќ
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ќ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_7408636p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ќ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љЮ
ѓ
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7411189
inputs_0
inputs_1	
inputs_29
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	А?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	А:
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	АA
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	А<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: G
4ecc_conv_2_fgn_out_tensordot_readvariableop_resource:	АA
2ecc_conv_2_fgn_out_biasadd_readvariableop_resource:	А<
*ecc_conv_2_shape_3_readvariableop_resource:  8
*ecc_conv_2_biasadd_readvariableop_resource: 9
&dense_1_matmul_readvariableop_resource:	 А6
'dense_1_biasadd_readvariableop_resource:	А:
&dense_2_matmul_readvariableop_resource:
АА6
'dense_2_biasadd_readvariableop_resource:	А8
%z_mean_matmul_readvariableop_resource:	А@4
&z_mean_biasadd_readvariableop_resource:@;
(z_log_var_matmul_readvariableop_resource:	А@7
)z_log_var_biasadd_readvariableop_resource:@*
 sampling_readvariableop_resource: /
%sampling_assignaddvariableop_resource: 1
'sampling_assignaddvariableop_1_resource: 1
'sampling_assignaddvariableop_2_resource: 1
'sampling_assignaddvariableop_3_resource: 
identity

identity_1ИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐecc_conv/BiasAdd/ReadVariableOpҐ'ecc_conv/FGN_out/BiasAdd/ReadVariableOpҐ)ecc_conv/FGN_out/Tensordot/ReadVariableOpҐ!ecc_conv/transpose/ReadVariableOpҐ!ecc_conv_1/BiasAdd/ReadVariableOpҐ)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpҐ+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpҐ#ecc_conv_1/transpose/ReadVariableOpҐ!ecc_conv_2/BiasAdd/ReadVariableOpҐ)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpҐ+ecc_conv_2/FGN_out/Tensordot/ReadVariableOpҐ#ecc_conv_2/transpose/ReadVariableOpҐsampling/AssignAddVariableOpҐsampling/AssignAddVariableOp_1Ґsampling/AssignAddVariableOp_2Ґsampling/AssignAddVariableOp_3Ґsampling/ReadVariableOpҐ"sampling/div_no_nan/ReadVariableOpҐ$sampling/div_no_nan/ReadVariableOp_1Ґ$sampling/div_no_nan_1/ReadVariableOpҐ&sampling/div_no_nan_1/ReadVariableOp_1Ґ z_log_var/BiasAdd/ReadVariableOpҐz_log_var/MatMul/ReadVariableOpҐz_mean/BiasAdd/ReadVariableOpҐz_mean/MatMul/ReadVariableOpr
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
graph_masking/strided_sliceStridedSliceinputs_0*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskt
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€v
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        v
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≠
graph_masking/strided_slice_1StridedSliceinputs_0,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskЖ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       i
dense/Tensordot/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ”
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : „
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : і
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense/Tensordot/transpose	Transpose$graph_masking/strided_slice:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<}
dense/leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=d
ecc_conv/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<g
ecc_conv/ShapeShape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:o
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€q
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€h
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
ecc_conv/Shape_1Shape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЭ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0i
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          X
 ecc_conv/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:j
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : €
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : а
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ђ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$ecc_conv/FGN_out/Tensordot/transpose	Transposeinputs_2*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<љ
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Њ
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аm
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аj
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ї
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АХ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0і
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аc
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Z
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ч
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Э
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< s
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            u
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            u
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ѓ
ecc_conv/strided_slice_2StridedSliceecc_conv/Cast:y:0'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskП
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< Ї
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0)dense/leaky_re_lu/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdi
ecc_conv/Shape_2Shape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:e
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numИ
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       e
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
numi
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Э
ecc_conv/Reshape_1Reshape)dense/leaky_re_lu/LeakyRelu:activations:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0h
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Х
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: i
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€Б
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: Е
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<\
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ≥
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:С
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Й
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Д
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0М
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< О
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ~
 ecc_conv/leaky_re_lu_1/LeakyRelu	LeakyReluecc_conv/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=f
ecc_conv_1/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<n
ecc_conv_1/ShapeShape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€s
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
ecc_conv_1/Shape_1Shape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:s
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€l
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask°
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0k
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:l
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:•
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeinputs_2,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<√
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ƒ
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аo
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аl
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АЩ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аe
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Г
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:£
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  u
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
ecc_conv_1/strided_slice_2StridedSliceecc_conv_1/Cast:y:0)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskХ
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  √
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdp
ecc_conv_1/Shape_2Shape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:i
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numМ
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0c
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        i
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numk
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
ecc_conv_1/Reshape_1Reshape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ О
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0j
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ы
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  k
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€З
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Л
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<^
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ї
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Ч
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< П
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< И
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Т
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< В
"ecc_conv_1/leaky_re_lu_2/LeakyRelu	LeakyReluecc_conv_1/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=f
ecc_conv_2/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<p
ecc_conv_2/ShapeShape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€s
 ecc_conv_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv_2/strided_sliceStridedSliceecc_conv_2/Shape:output:0'ecc_conv_2/strided_slice/stack:output:0)ecc_conv_2/strided_slice/stack_1:output:0)ecc_conv_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
ecc_conv_2/Shape_1Shape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
 ecc_conv_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€l
"ecc_conv_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"ecc_conv_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
ecc_conv_2/strided_slice_1StridedSliceecc_conv_2/Shape_1:output:0)ecc_conv_2/strided_slice_1/stack:output:0+ecc_conv_2/strided_slice_1/stack_1:output:0+ecc_conv_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask°
+ecc_conv_2/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_2_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0k
!ecc_conv_2/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!ecc_conv_2/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
"ecc_conv_2/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:l
*ecc_conv_2/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%ecc_conv_2/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_2/FGN_out/Tensordot/Shape:output:0*ecc_conv_2/FGN_out/Tensordot/free:output:03ecc_conv_2/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'ecc_conv_2/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_2/FGN_out/Tensordot/Shape:output:0*ecc_conv_2/FGN_out/Tensordot/axes:output:05ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"ecc_conv_2/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv_2/FGN_out/Tensordot/ProdProd.ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_2/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$ecc_conv_2/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#ecc_conv_2/FGN_out/Tensordot/Prod_1Prod0ecc_conv_2/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_2/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(ecc_conv_2/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#ecc_conv_2/FGN_out/Tensordot/concatConcatV2*ecc_conv_2/FGN_out/Tensordot/free:output:0*ecc_conv_2/FGN_out/Tensordot/axes:output:01ecc_conv_2/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"ecc_conv_2/FGN_out/Tensordot/stackPack*ecc_conv_2/FGN_out/Tensordot/Prod:output:0,ecc_conv_2/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:•
&ecc_conv_2/FGN_out/Tensordot/transpose	Transposeinputs_2,ecc_conv_2/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<√
$ecc_conv_2/FGN_out/Tensordot/ReshapeReshape*ecc_conv_2/FGN_out/Tensordot/transpose:y:0+ecc_conv_2/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ƒ
#ecc_conv_2/FGN_out/Tensordot/MatMulMatMul-ecc_conv_2/FGN_out/Tensordot/Reshape:output:03ecc_conv_2/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аo
$ecc_conv_2/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аl
*ecc_conv_2/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%ecc_conv_2/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_2/FGN_out/Tensordot/Const_2:output:03ecc_conv_2/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
ecc_conv_2/FGN_out/TensordotReshape-ecc_conv_2/FGN_out/Tensordot/MatMul:product:0.ecc_conv_2/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АЩ
)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_2_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ecc_conv_2/FGN_out/BiasAddBiasAdd%ecc_conv_2/FGN_out/Tensordot:output:01ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аe
ecc_conv_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
ecc_conv_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Г
ecc_conv_2/Reshape/shapePack#ecc_conv_2/Reshape/shape/0:output:0!ecc_conv_2/strided_slice:output:0!ecc_conv_2/strided_slice:output:0#ecc_conv_2/Reshape/shape/3:output:0#ecc_conv_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:£
ecc_conv_2/ReshapeReshape#ecc_conv_2/FGN_out/BiasAdd:output:0!ecc_conv_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  u
 ecc_conv_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
ecc_conv_2/strided_slice_2StridedSliceecc_conv_2/Cast:y:0)ecc_conv_2/strided_slice_2/stack:output:0+ecc_conv_2/strided_slice_2/stack_1:output:0+ecc_conv_2/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskХ
ecc_conv_2/mulMulecc_conv_2/Reshape:output:0#ecc_conv_2/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  ≈
ecc_conv_2/einsum/EinsumEinsumecc_conv_2/mul:z:00ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdr
ecc_conv_2/Shape_2Shape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:i
ecc_conv_2/unstackUnpackecc_conv_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numМ
!ecc_conv_2/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0c
ecc_conv_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        i
ecc_conv_2/unstack_1Unpackecc_conv_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
numk
ecc_conv_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ®
ecc_conv_2/Reshape_1Reshape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0#ecc_conv_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ О
#ecc_conv_2/transpose/ReadVariableOpReadVariableOp*ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0j
ecc_conv_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ы
ecc_conv_2/transpose	Transpose+ecc_conv_2/transpose/ReadVariableOp:value:0"ecc_conv_2/transpose/perm:output:0*
T0*
_output_shapes

:  k
ecc_conv_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€З
ecc_conv_2/Reshape_2Reshapeecc_conv_2/transpose:y:0#ecc_conv_2/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Л
ecc_conv_2/MatMulMatMulecc_conv_2/Reshape_1:output:0ecc_conv_2/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
ecc_conv_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<^
ecc_conv_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ї
ecc_conv_2/Reshape_3/shapePackecc_conv_2/unstack:output:0%ecc_conv_2/Reshape_3/shape/1:output:0%ecc_conv_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Ч
ecc_conv_2/Reshape_3Reshapeecc_conv_2/MatMul:product:0#ecc_conv_2/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< П
ecc_conv_2/addAddV2!ecc_conv_2/einsum/Einsum:output:0ecc_conv_2/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< И
!ecc_conv_2/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
ecc_conv_2/BiasAddBiasAddecc_conv_2/add:z:0)ecc_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Т
ecc_conv_2/mul_1Mulecc_conv_2/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< В
"ecc_conv_2/leaky_re_lu_3/LeakyRelu	LeakyReluecc_conv_2/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=p
%global_sum_pool/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€Ѓ
global_sum_pool/SumSum0ecc_conv_2/leaky_re_lu_3/LeakyRelu:activations:0.global_sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
dense_1/MatMulMatMulglobal_sum_pool/Sum:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dense_1/leaky_re_lu_4/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ф
flatten/ReshapeReshape-dense_1/leaky_re_lu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0М
dense_2/MatMulMatMulflatten/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dense_2/leaky_re_lu_5/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=Г
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Ю
z_mean/MatMulMatMul-dense_2/leaky_re_lu_5/LeakyRelu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0§
z_log_var/MatMulMatMul-dense_2/leaky_re_lu_5/LeakyRelu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@U
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?«
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Л’±
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?z
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@t
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@S
sampling/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?U
sampling/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
sampling/ReadVariableOpReadVariableOp sampling_readvariableop_resource*
_output_shapes
: *
dtype0U
sampling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<r
sampling/mul_2Mulsampling/mul_2/x:output:0sampling/ReadVariableOp:value:0*
T0*
_output_shapes
: S
sampling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>a
sampling/subSubsampling/mul_2:z:0sampling/sub/y:output:0*
T0*
_output_shapes
: i
sampling/MaximumMaximumsampling/Const_1:output:0sampling/sub:z:0*
T0*
_output_shapes
: k
sampling/MinimumMinimumsampling/Const:output:0sampling/Maximum:z:0*
T0*
_output_shapes
: U
sampling/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
sampling/add_1AddV2sampling/add_1/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
sampling/SquareSquarez_mean/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@p
sampling/sub_1Subsampling/add_1:z:0sampling/Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@c
sampling/Exp_1Expz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
sampling/sub_2Subsampling/sub_1:z:0sampling/Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@U
sampling/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њv
sampling/mul_3Mulsampling/mul_3/x:output:0sampling/sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@`
sampling/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :~
sampling/SumSumsampling/mul_3:z:0'sampling/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€Z
sampling/Const_2Const*
_output_shapes
:*
dtype0*
valueB: h
sampling/MeanMeansampling/Sum:output:0sampling/Const_2:output:0*
T0*
_output_shapes
: d
sampling/mul_4Mulsampling/Minimum:z:0sampling/Mean:output:0*
T0*
_output_shapes
: W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFm
sampling/truedivRealDivsampling/mul_4:z:0sampling/truediv/y:output:0*
T0*
_output_shapes
: O
sampling/RankConst*
_output_shapes
: *
dtype0*
value	B : V
sampling/range/startConst*
_output_shapes
: *
dtype0*
value	B : V
sampling/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :З
sampling/rangeRangesampling/range/start:output:0sampling/Rank:output:0sampling/range/delta:output:0*
_output_shapes
: g
sampling/Sum_1Sumsampling/Mean:output:0sampling/range:output:0*
T0*
_output_shapes
: Ц
sampling/AssignAddVariableOpAssignAddVariableOp%sampling_assignaddvariableop_resourcesampling/Sum_1:output:0*
_output_shapes
 *
dtype0O
sampling/SizeConst*
_output_shapes
: *
dtype0*
value	B :]
sampling/CastCastsampling/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ≥
sampling/AssignAddVariableOp_1AssignAddVariableOp'sampling_assignaddvariableop_1_resourcesampling/Cast:y:0^sampling/AssignAddVariableOp*
_output_shapes
 *
dtype0ј
"sampling/div_no_nan/ReadVariableOpReadVariableOp%sampling_assignaddvariableop_resource^sampling/AssignAddVariableOp^sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0•
$sampling/div_no_nan/ReadVariableOp_1ReadVariableOp'sampling_assignaddvariableop_1_resource^sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ъ
sampling/div_no_nanDivNoNan*sampling/div_no_nan/ReadVariableOp:value:0,sampling/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: W
sampling/IdentityIdentitysampling/div_no_nan:z:0*
T0*
_output_shapes
: Q
sampling/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : X
sampling/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : X
sampling/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :П
sampling/range_1Rangesampling/range_1/start:output:0sampling/Rank_1:output:0sampling/range_1/delta:output:0*
_output_shapes
: g
sampling/Sum_2Sumsampling/truediv:z:0sampling/range_1:output:0*
T0*
_output_shapes
: Ъ
sampling/AssignAddVariableOp_2AssignAddVariableOp'sampling_assignaddvariableop_2_resourcesampling/Sum_2:output:0*
_output_shapes
 *
dtype0Q
sampling/Size_1Const*
_output_shapes
: *
dtype0*
value	B :a
sampling/Cast_1Castsampling/Size_1:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
sampling/AssignAddVariableOp_3AssignAddVariableOp'sampling_assignaddvariableop_3_resourcesampling/Cast_1:y:0^sampling/AssignAddVariableOp_2*
_output_shapes
 *
dtype0∆
$sampling/div_no_nan_1/ReadVariableOpReadVariableOp'sampling_assignaddvariableop_2_resource^sampling/AssignAddVariableOp_2^sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0І
&sampling/div_no_nan_1/ReadVariableOp_1ReadVariableOp'sampling_assignaddvariableop_3_resource^sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0†
sampling/div_no_nan_1DivNoNan,sampling/div_no_nan_1/ReadVariableOp:value:0.sampling/div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
sampling/Identity_1Identitysampling/div_no_nan_1:z:0*
T0*
_output_shapes
: _
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@T

Identity_1Identitysampling/truediv:z:0^NoOp*
T0*
_output_shapes
: ®	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp"^ecc_conv_2/BiasAdd/ReadVariableOp*^ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_2/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_2/transpose/ReadVariableOp^sampling/AssignAddVariableOp^sampling/AssignAddVariableOp_1^sampling/AssignAddVariableOp_2^sampling/AssignAddVariableOp_3^sampling/ReadVariableOp#^sampling/div_no_nan/ReadVariableOp%^sampling/div_no_nan/ReadVariableOp_1%^sampling/div_no_nan_1/ReadVariableOp'^sampling/div_no_nan_1/ReadVariableOp_1!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp2F
!ecc_conv_2/BiasAdd/ReadVariableOp!ecc_conv_2/BiasAdd/ReadVariableOp2V
)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_2/FGN_out/Tensordot/ReadVariableOp+ecc_conv_2/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_2/transpose/ReadVariableOp#ecc_conv_2/transpose/ReadVariableOp2<
sampling/AssignAddVariableOpsampling/AssignAddVariableOp2@
sampling/AssignAddVariableOp_1sampling/AssignAddVariableOp_12@
sampling/AssignAddVariableOp_2sampling/AssignAddVariableOp_22@
sampling/AssignAddVariableOp_3sampling/AssignAddVariableOp_322
sampling/ReadVariableOpsampling/ReadVariableOp2H
"sampling/div_no_nan/ReadVariableOp"sampling/div_no_nan/ReadVariableOp2L
$sampling/div_no_nan/ReadVariableOp_1$sampling/div_no_nan/ReadVariableOp_12L
$sampling/div_no_nan_1/ReadVariableOp$sampling/div_no_nan_1/ReadVariableOp2P
&sampling/div_no_nan_1/ReadVariableOp_1&sampling/div_no_nan_1/ReadVariableOp_12D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
“	
ш
D__inference_dense_5_layer_call_and_return_conditional_losses_7408636

inputs2
matmul_readvariableop_resource:
АЌ.
biasadd_readvariableop_resource:	Ќ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ќw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ќ
ы
,__inference_ecc_conv_2_layer_call_fn_7411508
inputs_0
inputs_1	
inputs_2

mask_0
unknown:	А
	unknown_0:	А
	unknown_1:  
	unknown_2: 
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2mask_0unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€< 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
 
Ж
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7411232

inputs9
&dense_3_matmul_readvariableop_resource:	@А6
'dense_3_biasadd_readvariableop_resource:	А:
&dense_4_matmul_readvariableop_resource:
АІ6
'dense_4_biasadd_readvariableop_resource:	І:
&dense_5_matmul_readvariableop_resource:
АЌ6
'dense_5_biasadd_readvariableop_resource:	Ќ
identityИҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpЕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0z
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dense_3/leaky_re_lu_6/LeakyRelu	LeakyReludense_3/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=Ж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0°
dense_4/MatMulMatMul-dense_3/leaky_re_lu_6/LeakyRelu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Іg
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ІЖ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0°
dense_5/MatMulMatMul-dense_3/leaky_re_lu_6/LeakyRelu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЌГ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0П
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЌV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€У
concatConcatV2dense_4/Sigmoid:y:0dense_5/BiasAdd:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ф_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фЙ
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В
Ь
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645

inputs"
dense_3_7408604:	@А
dense_3_7408606:	А#
dense_4_7408621:
АІ
dense_4_7408623:	І#
dense_5_7408637:
АЌ
dense_5_7408639:	Ќ
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallу
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_7408604dense_3_7408606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_7408603Х
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_7408621dense_4_7408623*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€І*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_7408620Х
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_5_7408637dense_5_7408639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ќ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_7408636V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Є
concatConcatV2(dense_4/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ф_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
љЮ
ѓ
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7410823
inputs_0
inputs_1	
inputs_29
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	А?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	А:
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	АA
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	А<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: G
4ecc_conv_2_fgn_out_tensordot_readvariableop_resource:	АA
2ecc_conv_2_fgn_out_biasadd_readvariableop_resource:	А<
*ecc_conv_2_shape_3_readvariableop_resource:  8
*ecc_conv_2_biasadd_readvariableop_resource: 9
&dense_1_matmul_readvariableop_resource:	 А6
'dense_1_biasadd_readvariableop_resource:	А:
&dense_2_matmul_readvariableop_resource:
АА6
'dense_2_biasadd_readvariableop_resource:	А8
%z_mean_matmul_readvariableop_resource:	А@4
&z_mean_biasadd_readvariableop_resource:@;
(z_log_var_matmul_readvariableop_resource:	А@7
)z_log_var_biasadd_readvariableop_resource:@*
 sampling_readvariableop_resource: /
%sampling_assignaddvariableop_resource: 1
'sampling_assignaddvariableop_1_resource: 1
'sampling_assignaddvariableop_2_resource: 1
'sampling_assignaddvariableop_3_resource: 
identity

identity_1ИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐecc_conv/BiasAdd/ReadVariableOpҐ'ecc_conv/FGN_out/BiasAdd/ReadVariableOpҐ)ecc_conv/FGN_out/Tensordot/ReadVariableOpҐ!ecc_conv/transpose/ReadVariableOpҐ!ecc_conv_1/BiasAdd/ReadVariableOpҐ)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpҐ+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpҐ#ecc_conv_1/transpose/ReadVariableOpҐ!ecc_conv_2/BiasAdd/ReadVariableOpҐ)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpҐ+ecc_conv_2/FGN_out/Tensordot/ReadVariableOpҐ#ecc_conv_2/transpose/ReadVariableOpҐsampling/AssignAddVariableOpҐsampling/AssignAddVariableOp_1Ґsampling/AssignAddVariableOp_2Ґsampling/AssignAddVariableOp_3Ґsampling/ReadVariableOpҐ"sampling/div_no_nan/ReadVariableOpҐ$sampling/div_no_nan/ReadVariableOp_1Ґ$sampling/div_no_nan_1/ReadVariableOpҐ&sampling/div_no_nan_1/ReadVariableOp_1Ґ z_log_var/BiasAdd/ReadVariableOpҐz_log_var/MatMul/ReadVariableOpҐz_mean/BiasAdd/ReadVariableOpҐz_mean/MatMul/ReadVariableOpr
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
graph_masking/strided_sliceStridedSliceinputs_0*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskt
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€v
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        v
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≠
graph_masking/strided_slice_1StridedSliceinputs_0,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskЖ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       i
dense/Tensordot/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ”
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : „
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : і
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense/Tensordot/transpose	Transpose$graph_masking/strided_slice:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<}
dense/leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=d
ecc_conv/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<g
ecc_conv/ShapeShape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:o
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€q
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€h
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
ecc_conv/Shape_1Shape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЭ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0i
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          X
 ecc_conv/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:j
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : €
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : а
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ђ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:°
$ecc_conv/FGN_out/Tensordot/transpose	Transposeinputs_2*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<љ
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Њ
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аm
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аj
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ї
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АХ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0і
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аc
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Z
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ч
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Э
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< s
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            u
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            u
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ѓ
ecc_conv/strided_slice_2StridedSliceecc_conv/Cast:y:0'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskП
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< Ї
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0)dense/leaky_re_lu/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdi
ecc_conv/Shape_2Shape)dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:e
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numИ
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       e
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
numi
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Э
ecc_conv/Reshape_1Reshape)dense/leaky_re_lu/LeakyRelu:activations:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0h
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Х
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: i
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€Б
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: Е
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<\
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ≥
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:С
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Й
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Д
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0М
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< О
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ~
 ecc_conv/leaky_re_lu_1/LeakyRelu	LeakyReluecc_conv/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=f
ecc_conv_1/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<n
ecc_conv_1/ShapeShape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€s
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
ecc_conv_1/Shape_1Shape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:s
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€l
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask°
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0k
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:l
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:•
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeinputs_2,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<√
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ƒ
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аo
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аl
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АЩ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аe
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Г
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:£
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  u
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
ecc_conv_1/strided_slice_2StridedSliceecc_conv_1/Cast:y:0)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskХ
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  √
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdp
ecc_conv_1/Shape_2Shape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:i
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numМ
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0c
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        i
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numk
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
ecc_conv_1/Reshape_1Reshape.ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ О
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0j
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ы
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  k
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€З
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Л
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<^
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ї
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Ч
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< П
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< И
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Т
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< В
"ecc_conv_1/leaky_re_lu_2/LeakyRelu	LeakyReluecc_conv_1/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=f
ecc_conv_2/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<p
ecc_conv_2/ShapeShape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:q
ecc_conv_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€s
 ecc_conv_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 ecc_conv_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
ecc_conv_2/strided_sliceStridedSliceecc_conv_2/Shape:output:0'ecc_conv_2/strided_slice/stack:output:0)ecc_conv_2/strided_slice/stack_1:output:0)ecc_conv_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
ecc_conv_2/Shape_1Shape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
 ecc_conv_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€l
"ecc_conv_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"ecc_conv_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
ecc_conv_2/strided_slice_1StridedSliceecc_conv_2/Shape_1:output:0)ecc_conv_2/strided_slice_1/stack:output:0+ecc_conv_2/strided_slice_1/stack_1:output:0+ecc_conv_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask°
+ecc_conv_2/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_2_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0k
!ecc_conv_2/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!ecc_conv_2/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
"ecc_conv_2/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:l
*ecc_conv_2/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%ecc_conv_2/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_2/FGN_out/Tensordot/Shape:output:0*ecc_conv_2/FGN_out/Tensordot/free:output:03ecc_conv_2/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'ecc_conv_2/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_2/FGN_out/Tensordot/Shape:output:0*ecc_conv_2/FGN_out/Tensordot/axes:output:05ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"ecc_conv_2/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!ecc_conv_2/FGN_out/Tensordot/ProdProd.ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_2/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$ecc_conv_2/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#ecc_conv_2/FGN_out/Tensordot/Prod_1Prod0ecc_conv_2/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_2/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(ecc_conv_2/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#ecc_conv_2/FGN_out/Tensordot/concatConcatV2*ecc_conv_2/FGN_out/Tensordot/free:output:0*ecc_conv_2/FGN_out/Tensordot/axes:output:01ecc_conv_2/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"ecc_conv_2/FGN_out/Tensordot/stackPack*ecc_conv_2/FGN_out/Tensordot/Prod:output:0,ecc_conv_2/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:•
&ecc_conv_2/FGN_out/Tensordot/transpose	Transposeinputs_2,ecc_conv_2/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<√
$ecc_conv_2/FGN_out/Tensordot/ReshapeReshape*ecc_conv_2/FGN_out/Tensordot/transpose:y:0+ecc_conv_2/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ƒ
#ecc_conv_2/FGN_out/Tensordot/MatMulMatMul-ecc_conv_2/FGN_out/Tensordot/Reshape:output:03ecc_conv_2/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аo
$ecc_conv_2/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аl
*ecc_conv_2/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%ecc_conv_2/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_2/FGN_out/Tensordot/Const_2:output:03ecc_conv_2/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
ecc_conv_2/FGN_out/TensordotReshape-ecc_conv_2/FGN_out/Tensordot/MatMul:product:0.ecc_conv_2/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АЩ
)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_2_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ecc_conv_2/FGN_out/BiasAddBiasAdd%ecc_conv_2/FGN_out/Tensordot:output:01ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аe
ecc_conv_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
ecc_conv_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Г
ecc_conv_2/Reshape/shapePack#ecc_conv_2/Reshape/shape/0:output:0!ecc_conv_2/strided_slice:output:0!ecc_conv_2/strided_slice:output:0#ecc_conv_2/Reshape/shape/3:output:0#ecc_conv_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:£
ecc_conv_2/ReshapeReshape#ecc_conv_2/FGN_out/BiasAdd:output:0!ecc_conv_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  u
 ecc_conv_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"ecc_conv_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
ecc_conv_2/strided_slice_2StridedSliceecc_conv_2/Cast:y:0)ecc_conv_2/strided_slice_2/stack:output:0+ecc_conv_2/strided_slice_2/stack_1:output:0+ecc_conv_2/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskХ
ecc_conv_2/mulMulecc_conv_2/Reshape:output:0#ecc_conv_2/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  ≈
ecc_conv_2/einsum/EinsumEinsumecc_conv_2/mul:z:00ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdr
ecc_conv_2/Shape_2Shape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:i
ecc_conv_2/unstackUnpackecc_conv_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numМ
!ecc_conv_2/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0c
ecc_conv_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        i
ecc_conv_2/unstack_1Unpackecc_conv_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
numk
ecc_conv_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ®
ecc_conv_2/Reshape_1Reshape0ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0#ecc_conv_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ О
#ecc_conv_2/transpose/ReadVariableOpReadVariableOp*ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0j
ecc_conv_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ы
ecc_conv_2/transpose	Transpose+ecc_conv_2/transpose/ReadVariableOp:value:0"ecc_conv_2/transpose/perm:output:0*
T0*
_output_shapes

:  k
ecc_conv_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€З
ecc_conv_2/Reshape_2Reshapeecc_conv_2/transpose:y:0#ecc_conv_2/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Л
ecc_conv_2/MatMulMatMulecc_conv_2/Reshape_1:output:0ecc_conv_2/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
ecc_conv_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<^
ecc_conv_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ї
ecc_conv_2/Reshape_3/shapePackecc_conv_2/unstack:output:0%ecc_conv_2/Reshape_3/shape/1:output:0%ecc_conv_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Ч
ecc_conv_2/Reshape_3Reshapeecc_conv_2/MatMul:product:0#ecc_conv_2/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< П
ecc_conv_2/addAddV2!ecc_conv_2/einsum/Einsum:output:0ecc_conv_2/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< И
!ecc_conv_2/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
ecc_conv_2/BiasAddBiasAddecc_conv_2/add:z:0)ecc_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Т
ecc_conv_2/mul_1Mulecc_conv_2/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< В
"ecc_conv_2/leaky_re_lu_3/LeakyRelu	LeakyReluecc_conv_2/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=p
%global_sum_pool/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€Ѓ
global_sum_pool/SumSum0ecc_conv_2/leaky_re_lu_3/LeakyRelu:activations:0.global_sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
dense_1/MatMulMatMulglobal_sum_pool/Sum:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dense_1/leaky_re_lu_4/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ф
flatten/ReshapeReshape-dense_1/leaky_re_lu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0М
dense_2/MatMulMatMulflatten/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
dense_2/leaky_re_lu_5/LeakyRelu	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=Г
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Ю
z_mean/MatMulMatMul-dense_2/leaky_re_lu_5/LeakyRelu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0§
z_log_var/MatMulMatMul-dense_2/leaky_re_lu_5/LeakyRelu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@U
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?«
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2эЈI±
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?z
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@t
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@S
sampling/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?U
sampling/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    p
sampling/ReadVariableOpReadVariableOp sampling_readvariableop_resource*
_output_shapes
: *
dtype0U
sampling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<r
sampling/mul_2Mulsampling/mul_2/x:output:0sampling/ReadVariableOp:value:0*
T0*
_output_shapes
: S
sampling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>a
sampling/subSubsampling/mul_2:z:0sampling/sub/y:output:0*
T0*
_output_shapes
: i
sampling/MaximumMaximumsampling/Const_1:output:0sampling/sub:z:0*
T0*
_output_shapes
: k
sampling/MinimumMinimumsampling/Const:output:0sampling/Maximum:z:0*
T0*
_output_shapes
: U
sampling/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
sampling/add_1AddV2sampling/add_1/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
sampling/SquareSquarez_mean/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@p
sampling/sub_1Subsampling/add_1:z:0sampling/Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@c
sampling/Exp_1Expz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
sampling/sub_2Subsampling/sub_1:z:0sampling/Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@U
sampling/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њv
sampling/mul_3Mulsampling/mul_3/x:output:0sampling/sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@`
sampling/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :~
sampling/SumSumsampling/mul_3:z:0'sampling/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€Z
sampling/Const_2Const*
_output_shapes
:*
dtype0*
valueB: h
sampling/MeanMeansampling/Sum:output:0sampling/Const_2:output:0*
T0*
_output_shapes
: d
sampling/mul_4Mulsampling/Minimum:z:0sampling/Mean:output:0*
T0*
_output_shapes
: W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFm
sampling/truedivRealDivsampling/mul_4:z:0sampling/truediv/y:output:0*
T0*
_output_shapes
: O
sampling/RankConst*
_output_shapes
: *
dtype0*
value	B : V
sampling/range/startConst*
_output_shapes
: *
dtype0*
value	B : V
sampling/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :З
sampling/rangeRangesampling/range/start:output:0sampling/Rank:output:0sampling/range/delta:output:0*
_output_shapes
: g
sampling/Sum_1Sumsampling/Mean:output:0sampling/range:output:0*
T0*
_output_shapes
: Ц
sampling/AssignAddVariableOpAssignAddVariableOp%sampling_assignaddvariableop_resourcesampling/Sum_1:output:0*
_output_shapes
 *
dtype0O
sampling/SizeConst*
_output_shapes
: *
dtype0*
value	B :]
sampling/CastCastsampling/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ≥
sampling/AssignAddVariableOp_1AssignAddVariableOp'sampling_assignaddvariableop_1_resourcesampling/Cast:y:0^sampling/AssignAddVariableOp*
_output_shapes
 *
dtype0ј
"sampling/div_no_nan/ReadVariableOpReadVariableOp%sampling_assignaddvariableop_resource^sampling/AssignAddVariableOp^sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0•
$sampling/div_no_nan/ReadVariableOp_1ReadVariableOp'sampling_assignaddvariableop_1_resource^sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ъ
sampling/div_no_nanDivNoNan*sampling/div_no_nan/ReadVariableOp:value:0,sampling/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: W
sampling/IdentityIdentitysampling/div_no_nan:z:0*
T0*
_output_shapes
: Q
sampling/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : X
sampling/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : X
sampling/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :П
sampling/range_1Rangesampling/range_1/start:output:0sampling/Rank_1:output:0sampling/range_1/delta:output:0*
_output_shapes
: g
sampling/Sum_2Sumsampling/truediv:z:0sampling/range_1:output:0*
T0*
_output_shapes
: Ъ
sampling/AssignAddVariableOp_2AssignAddVariableOp'sampling_assignaddvariableop_2_resourcesampling/Sum_2:output:0*
_output_shapes
 *
dtype0Q
sampling/Size_1Const*
_output_shapes
: *
dtype0*
value	B :a
sampling/Cast_1Castsampling/Size_1:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
sampling/AssignAddVariableOp_3AssignAddVariableOp'sampling_assignaddvariableop_3_resourcesampling/Cast_1:y:0^sampling/AssignAddVariableOp_2*
_output_shapes
 *
dtype0∆
$sampling/div_no_nan_1/ReadVariableOpReadVariableOp'sampling_assignaddvariableop_2_resource^sampling/AssignAddVariableOp_2^sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0І
&sampling/div_no_nan_1/ReadVariableOp_1ReadVariableOp'sampling_assignaddvariableop_3_resource^sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0†
sampling/div_no_nan_1DivNoNan,sampling/div_no_nan_1/ReadVariableOp:value:0.sampling/div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
sampling/Identity_1Identitysampling/div_no_nan_1:z:0*
T0*
_output_shapes
: _
IdentityIdentitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@T

Identity_1Identitysampling/truediv:z:0^NoOp*
T0*
_output_shapes
: ®	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp"^ecc_conv_2/BiasAdd/ReadVariableOp*^ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_2/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_2/transpose/ReadVariableOp^sampling/AssignAddVariableOp^sampling/AssignAddVariableOp_1^sampling/AssignAddVariableOp_2^sampling/AssignAddVariableOp_3^sampling/ReadVariableOp#^sampling/div_no_nan/ReadVariableOp%^sampling/div_no_nan/ReadVariableOp_1%^sampling/div_no_nan_1/ReadVariableOp'^sampling/div_no_nan_1/ReadVariableOp_1!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp2F
!ecc_conv_2/BiasAdd/ReadVariableOp!ecc_conv_2/BiasAdd/ReadVariableOp2V
)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_2/FGN_out/Tensordot/ReadVariableOp+ecc_conv_2/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_2/transpose/ReadVariableOp#ecc_conv_2/transpose/ReadVariableOp2<
sampling/AssignAddVariableOpsampling/AssignAddVariableOp2@
sampling/AssignAddVariableOp_1sampling/AssignAddVariableOp_12@
sampling/AssignAddVariableOp_2sampling/AssignAddVariableOp_22@
sampling/AssignAddVariableOp_3sampling/AssignAddVariableOp_322
sampling/ReadVariableOpsampling/ReadVariableOp2H
"sampling/div_no_nan/ReadVariableOp"sampling/div_no_nan/ReadVariableOp2L
$sampling/div_no_nan/ReadVariableOp_1$sampling/div_no_nan/ReadVariableOp_12L
$sampling/div_no_nan_1/ReadVariableOp$sampling/div_no_nan_1/ReadVariableOp2P
&sampling/div_no_nan_1/ReadVariableOp_1&sampling/div_no_nan_1/ReadVariableOp_12D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
∆
Ц
(__inference_z_mean_layer_call_fn_7411660

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
цШ
“!
@__inference_vae_layer_call_and_return_conditional_losses_7409945
inputs_0
inputs_1	
inputs_2F
4nyan_encoder_dense_tensordot_readvariableop_resource:@
2nyan_encoder_dense_biasadd_readvariableop_resource:R
?nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource:	АL
=nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource:	АG
5nyan_encoder_ecc_conv_shape_3_readvariableop_resource: C
5nyan_encoder_ecc_conv_biasadd_readvariableop_resource: T
Anyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	АN
?nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	АI
7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource:  E
7nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource: T
Anyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource:	АN
?nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource:	АI
7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource:  E
7nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource: F
3nyan_encoder_dense_1_matmul_readvariableop_resource:	 АC
4nyan_encoder_dense_1_biasadd_readvariableop_resource:	АG
3nyan_encoder_dense_2_matmul_readvariableop_resource:
ААC
4nyan_encoder_dense_2_biasadd_readvariableop_resource:	АE
2nyan_encoder_z_mean_matmul_readvariableop_resource:	А@A
3nyan_encoder_z_mean_biasadd_readvariableop_resource:@H
5nyan_encoder_z_log_var_matmul_readvariableop_resource:	А@D
6nyan_encoder_z_log_var_biasadd_readvariableop_resource:@7
-nyan_encoder_sampling_readvariableop_resource: <
2nyan_encoder_sampling_assignaddvariableop_resource: >
4nyan_encoder_sampling_assignaddvariableop_1_resource: >
4nyan_encoder_sampling_assignaddvariableop_2_resource: >
4nyan_encoder_sampling_assignaddvariableop_3_resource: F
3nyan_decoder_dense_3_matmul_readvariableop_resource:	@АC
4nyan_decoder_dense_3_biasadd_readvariableop_resource:	АG
3nyan_decoder_dense_4_matmul_readvariableop_resource:
АІC
4nyan_decoder_dense_4_biasadd_readvariableop_resource:	ІG
3nyan_decoder_dense_5_matmul_readvariableop_resource:
АЌC
4nyan_decoder_dense_5_biasadd_readvariableop_resource:	Ќ
identity

identity_1ИҐ+nyan_decoder/dense_3/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_3/MatMul/ReadVariableOpҐ+nyan_decoder/dense_4/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_4/MatMul/ReadVariableOpҐ+nyan_decoder/dense_5/BiasAdd/ReadVariableOpҐ*nyan_decoder/dense_5/MatMul/ReadVariableOpҐ)nyan_encoder/dense/BiasAdd/ReadVariableOpҐ+nyan_encoder/dense/Tensordot/ReadVariableOpҐ+nyan_encoder/dense_1/BiasAdd/ReadVariableOpҐ*nyan_encoder/dense_1/MatMul/ReadVariableOpҐ+nyan_encoder/dense_2/BiasAdd/ReadVariableOpҐ*nyan_encoder/dense_2/MatMul/ReadVariableOpҐ,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpҐ4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpҐ.nyan_encoder/ecc_conv/transpose/ReadVariableOpҐ.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpҐ8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpҐ0nyan_encoder/ecc_conv_1/transpose/ReadVariableOpҐ.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpҐ6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpҐ8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpҐ0nyan_encoder/ecc_conv_2/transpose/ReadVariableOpҐ)nyan_encoder/sampling/AssignAddVariableOpҐ+nyan_encoder/sampling/AssignAddVariableOp_1Ґ+nyan_encoder/sampling/AssignAddVariableOp_2Ґ+nyan_encoder/sampling/AssignAddVariableOp_3Ґ$nyan_encoder/sampling/ReadVariableOpҐ/nyan_encoder/sampling/div_no_nan/ReadVariableOpҐ1nyan_encoder/sampling/div_no_nan/ReadVariableOp_1Ґ1nyan_encoder/sampling/div_no_nan_1/ReadVariableOpҐ3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1Ґ-nyan_encoder/z_log_var/BiasAdd/ReadVariableOpҐ,nyan_encoder/z_log_var/MatMul/ReadVariableOpҐ*nyan_encoder/z_mean/BiasAdd/ReadVariableOpҐ)nyan_encoder/z_mean/MatMul/ReadVariableOp
.nyan_encoder/graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Б
0nyan_encoder/graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€Б
0nyan_encoder/graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      џ
(nyan_encoder/graph_masking/strided_sliceStridedSliceinputs_07nyan_encoder/graph_masking/strided_slice/stack:output:09nyan_encoder/graph_masking/strided_slice/stack_1:output:09nyan_encoder/graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskБ
0nyan_encoder/graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Г
2nyan_encoder/graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Г
2nyan_encoder/graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      б
*nyan_encoder/graph_masking/strided_slice_1StridedSliceinputs_09nyan_encoder/graph_masking/strided_slice_1/stack:output:0;nyan_encoder/graph_masking/strided_slice_1/stack_1:output:0;nyan_encoder/graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_mask†
+nyan_encoder/dense/Tensordot/ReadVariableOpReadVariableOp4nyan_encoder_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!nyan_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!nyan_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Г
"nyan_encoder/dense/Tensordot/ShapeShape1nyan_encoder/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:l
*nyan_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : З
%nyan_encoder/dense/Tensordot/GatherV2GatherV2+nyan_encoder/dense/Tensordot/Shape:output:0*nyan_encoder/dense/Tensordot/free:output:03nyan_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,nyan_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
'nyan_encoder/dense/Tensordot/GatherV2_1GatherV2+nyan_encoder/dense/Tensordot/Shape:output:0*nyan_encoder/dense/Tensordot/axes:output:05nyan_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"nyan_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: І
!nyan_encoder/dense/Tensordot/ProdProd.nyan_encoder/dense/Tensordot/GatherV2:output:0+nyan_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$nyan_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≠
#nyan_encoder/dense/Tensordot/Prod_1Prod0nyan_encoder/dense/Tensordot/GatherV2_1:output:0-nyan_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(nyan_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
#nyan_encoder/dense/Tensordot/concatConcatV2*nyan_encoder/dense/Tensordot/free:output:0*nyan_encoder/dense/Tensordot/axes:output:01nyan_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:≤
"nyan_encoder/dense/Tensordot/stackPack*nyan_encoder/dense/Tensordot/Prod:output:0,nyan_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
&nyan_encoder/dense/Tensordot/transpose	Transpose1nyan_encoder/graph_masking/strided_slice:output:0,nyan_encoder/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<√
$nyan_encoder/dense/Tensordot/ReshapeReshape*nyan_encoder/dense/Tensordot/transpose:y:0+nyan_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€√
#nyan_encoder/dense/Tensordot/MatMulMatMul-nyan_encoder/dense/Tensordot/Reshape:output:03nyan_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€n
$nyan_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*nyan_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
%nyan_encoder/dense/Tensordot/concat_1ConcatV2.nyan_encoder/dense/Tensordot/GatherV2:output:0-nyan_encoder/dense/Tensordot/Const_2:output:03nyan_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Љ
nyan_encoder/dense/TensordotReshape-nyan_encoder/dense/Tensordot/MatMul:product:0.nyan_encoder/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<Ш
)nyan_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2nyan_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
nyan_encoder/dense/BiasAddBiasAdd%nyan_encoder/dense/Tensordot:output:01nyan_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<Ч
(nyan_encoder/dense/leaky_re_lu/LeakyRelu	LeakyRelu#nyan_encoder/dense/BiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=q
nyan_encoder/ecc_conv/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<Б
nyan_encoder/ecc_conv/ShapeShape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:|
)nyan_encoder/ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€~
+nyan_encoder/ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€u
+nyan_encoder/ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#nyan_encoder/ecc_conv/strided_sliceStridedSlice$nyan_encoder/ecc_conv/Shape:output:02nyan_encoder/ecc_conv/strided_slice/stack:output:04nyan_encoder/ecc_conv/strided_slice/stack_1:output:04nyan_encoder/ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskГ
nyan_encoder/ecc_conv/Shape_1Shape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-nyan_encoder/ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv/strided_slice_1StridedSlice&nyan_encoder/ecc_conv/Shape_1:output:04nyan_encoder/ecc_conv/strided_slice_1/stack:output:06nyan_encoder/ecc_conv/strided_slice_1/stack_1:output:06nyan_encoder/ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЈ
6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0v
,nyan_encoder/ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Б
,nyan_encoder/ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          e
-nyan_encoder/ecc_conv/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:w
5nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ≥
0nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2GatherV26nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
2nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV26nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0@nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-nyan_encoder/ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: »
,nyan_encoder/ecc_conv/FGN_out/Tensordot/ProdProd9nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:06nyan_encoder/ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1Prod;nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1:output:08nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.nyan_encoder/ecc_conv/FGN_out/Tensordot/concatConcatV25nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:05nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0<nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:”
-nyan_encoder/ecc_conv/FGN_out/Tensordot/stackPack5nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod:output:07nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ї
1nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose	Transposeinputs_27nyan_encoder/ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<д
/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReshapeReshape5nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose:y:06nyan_encoder/ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€е
.nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMulMatMul8nyan_encoder/ecc_conv/FGN_out/Tensordot/Reshape:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аz
/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аw
5nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1ConcatV29nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2:output:0>nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:в
'nyan_encoder/ecc_conv/FGN_out/TensordotReshape8nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMul:product:09nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аѓ
4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp=nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0џ
%nyan_encoder/ecc_conv/FGN_out/BiasAddBiasAdd0nyan_encoder/ecc_conv/FGN_out/Tensordot:output:0<nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аp
%nyan_encoder/ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€g
%nyan_encoder/ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ≈
#nyan_encoder/ecc_conv/Reshape/shapePack.nyan_encoder/ecc_conv/Reshape/shape/0:output:0,nyan_encoder/ecc_conv/strided_slice:output:0,nyan_encoder/ecc_conv/strided_slice:output:0.nyan_encoder/ecc_conv/Reshape/shape/3:output:0.nyan_encoder/ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ƒ
nyan_encoder/ecc_conv/ReshapeReshape.nyan_encoder/ecc_conv/FGN_out/BiasAdd:output:0,nyan_encoder/ecc_conv/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< А
+nyan_encoder/ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            В
-nyan_encoder/ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            В
-nyan_encoder/ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         р
%nyan_encoder/ecc_conv/strided_slice_2StridedSlicenyan_encoder/ecc_conv/Cast:y:04nyan_encoder/ecc_conv/strided_slice_2/stack:output:06nyan_encoder/ecc_conv/strided_slice_2/stack_1:output:06nyan_encoder/ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskґ
nyan_encoder/ecc_conv/mulMul&nyan_encoder/ecc_conv/Reshape:output:0.nyan_encoder/ecc_conv/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< б
#nyan_encoder/ecc_conv/einsum/EinsumEinsumnyan_encoder/ecc_conv/mul:z:06nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdГ
nyan_encoder/ecc_conv/Shape_2Shape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:
nyan_encoder/ecc_conv/unstackUnpack&nyan_encoder/ecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numҐ
,nyan_encoder/ecc_conv/Shape_3/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0n
nyan_encoder/ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       
nyan_encoder/ecc_conv/unstack_1Unpack&nyan_encoder/ecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
numv
%nyan_encoder/ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ƒ
nyan_encoder/ecc_conv/Reshape_1Reshape6nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0.nyan_encoder/ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€§
.nyan_encoder/ecc_conv/transpose/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0u
$nyan_encoder/ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Љ
nyan_encoder/ecc_conv/transpose	Transpose6nyan_encoder/ecc_conv/transpose/ReadVariableOp:value:0-nyan_encoder/ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: v
%nyan_encoder/ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€®
nyan_encoder/ecc_conv/Reshape_2Reshape#nyan_encoder/ecc_conv/transpose:y:0.nyan_encoder/ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: ђ
nyan_encoder/ecc_conv/MatMulMatMul(nyan_encoder/ecc_conv/Reshape_1:output:0(nyan_encoder/ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ i
'nyan_encoder/ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<i
'nyan_encoder/ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : з
%nyan_encoder/ecc_conv/Reshape_3/shapePack&nyan_encoder/ecc_conv/unstack:output:00nyan_encoder/ecc_conv/Reshape_3/shape/1:output:00nyan_encoder/ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Є
nyan_encoder/ecc_conv/Reshape_3Reshape&nyan_encoder/ecc_conv/MatMul:product:0.nyan_encoder/ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ∞
nyan_encoder/ecc_conv/addAddV2,nyan_encoder/ecc_conv/einsum/Einsum:output:0(nyan_encoder/ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ю
,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpReadVariableOp5nyan_encoder_ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≥
nyan_encoder/ecc_conv/BiasAddBiasAddnyan_encoder/ecc_conv/add:z:04nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< µ
nyan_encoder/ecc_conv/mul_1Mul&nyan_encoder/ecc_conv/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ш
-nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu	LeakyRelunyan_encoder/ecc_conv/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=s
nyan_encoder/ecc_conv_1/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<И
nyan_encoder/ecc_conv_1/ShapeShape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€А
-nyan_encoder/ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv_1/strided_sliceStridedSlice&nyan_encoder/ecc_conv_1/Shape:output:04nyan_encoder/ecc_conv_1/strided_slice/stack:output:06nyan_encoder/ecc_conv_1/strided_slice/stack_1:output:06nyan_encoder/ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
nyan_encoder/ecc_conv_1/Shape_1Shape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:А
-nyan_encoder/ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€y
/nyan_encoder/ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/nyan_encoder/ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'nyan_encoder/ecc_conv_1/strided_slice_1StridedSlice(nyan_encoder/ecc_conv_1/Shape_1:output:06nyan_encoder/ecc_conv_1/strided_slice_1/stack:output:08nyan_encoder/ecc_conv_1/strided_slice_1/stack_1:output:08nyan_encoder/ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOpAnyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0x
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Г
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          g
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:y
7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
2nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV28nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
4nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV28nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0Bnyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ProdProd;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ‘
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1Prod=nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concatConcatV27nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:07nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0>nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stackPack7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod:output:09nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:њ
3nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose	Transposeinputs_29nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<к
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReshapeReshape7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose:y:08nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€л
0nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMulMatMul:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Reshape:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А|
1nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аy
7nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
2nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2:output:0@nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
)nyan_encoder/ecc_conv_1/FGN_out/TensordotReshape:nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMul:product:0;nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<А≥
6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
'nyan_encoder/ecc_conv_1/FGN_out/BiasAddBiasAdd2nyan_encoder/ecc_conv_1/FGN_out/Tensordot:output:0>nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аr
'nyan_encoder/ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€i
'nyan_encoder/ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : —
%nyan_encoder/ecc_conv_1/Reshape/shapePack0nyan_encoder/ecc_conv_1/Reshape/shape/0:output:0.nyan_encoder/ecc_conv_1/strided_slice:output:0.nyan_encoder/ecc_conv_1/strided_slice:output:00nyan_encoder/ecc_conv_1/Reshape/shape/3:output:00nyan_encoder/ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
: 
nyan_encoder/ecc_conv_1/ReshapeReshape0nyan_encoder/ecc_conv_1/FGN_out/BiasAdd:output:0.nyan_encoder/ecc_conv_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  В
-nyan_encoder/ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ъ
'nyan_encoder/ecc_conv_1/strided_slice_2StridedSlice nyan_encoder/ecc_conv_1/Cast:y:06nyan_encoder/ecc_conv_1/strided_slice_2/stack:output:08nyan_encoder/ecc_conv_1/strided_slice_2/stack_1:output:08nyan_encoder/ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskЉ
nyan_encoder/ecc_conv_1/mulMul(nyan_encoder/ecc_conv_1/Reshape:output:00nyan_encoder/ecc_conv_1/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  к
%nyan_encoder/ecc_conv_1/einsum/EinsumEinsumnyan_encoder/ecc_conv_1/mul:z:0;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdК
nyan_encoder/ecc_conv_1/Shape_2Shape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:Г
nyan_encoder/ecc_conv_1/unstackUnpack(nyan_encoder/ecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num¶
.nyan_encoder/ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0p
nyan_encoder/ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Г
!nyan_encoder/ecc_conv_1/unstack_1Unpack(nyan_encoder/ecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numx
'nyan_encoder/ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ќ
!nyan_encoder/ecc_conv_1/Reshape_1Reshape;nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:00nyan_encoder/ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ®
0nyan_encoder/ecc_conv_1/transpose/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0w
&nyan_encoder/ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!nyan_encoder/ecc_conv_1/transpose	Transpose8nyan_encoder/ecc_conv_1/transpose/ReadVariableOp:value:0/nyan_encoder/ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  x
'nyan_encoder/ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ѓ
!nyan_encoder/ecc_conv_1/Reshape_2Reshape%nyan_encoder/ecc_conv_1/transpose:y:00nyan_encoder/ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  ≤
nyan_encoder/ecc_conv_1/MatMulMatMul*nyan_encoder/ecc_conv_1/Reshape_1:output:0*nyan_encoder/ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
)nyan_encoder/ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<k
)nyan_encoder/ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
'nyan_encoder/ecc_conv_1/Reshape_3/shapePack(nyan_encoder/ecc_conv_1/unstack:output:02nyan_encoder/ecc_conv_1/Reshape_3/shape/1:output:02nyan_encoder/ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Њ
!nyan_encoder/ecc_conv_1/Reshape_3Reshape(nyan_encoder/ecc_conv_1/MatMul:product:00nyan_encoder/ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ґ
nyan_encoder/ecc_conv_1/addAddV2.nyan_encoder/ecc_conv_1/einsum/Einsum:output:0*nyan_encoder/ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ґ
.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
nyan_encoder/ecc_conv_1/BiasAddBiasAddnyan_encoder/ecc_conv_1/add:z:06nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< є
nyan_encoder/ecc_conv_1/mul_1Mul(nyan_encoder/ecc_conv_1/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ь
/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu	LeakyRelu!nyan_encoder/ecc_conv_1/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=s
nyan_encoder/ecc_conv_2/CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<К
nyan_encoder/ecc_conv_2/ShapeShape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:~
+nyan_encoder/ecc_conv_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€А
-nyan_encoder/ecc_conv_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-nyan_encoder/ecc_conv_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/ecc_conv_2/strided_sliceStridedSlice&nyan_encoder/ecc_conv_2/Shape:output:04nyan_encoder/ecc_conv_2/strided_slice/stack:output:06nyan_encoder/ecc_conv_2/strided_slice/stack_1:output:06nyan_encoder/ecc_conv_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
nyan_encoder/ecc_conv_2/Shape_1Shape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:А
-nyan_encoder/ecc_conv_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€y
/nyan_encoder/ecc_conv_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/nyan_encoder/ecc_conv_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'nyan_encoder/ecc_conv_2/strided_slice_1StridedSlice(nyan_encoder/ecc_conv_2/Shape_1:output:06nyan_encoder/ecc_conv_2/strided_slice_1/stack:output:08nyan_encoder/ecc_conv_2/strided_slice_1/stack_1:output:08nyan_encoder/ecc_conv_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpReadVariableOpAnyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0x
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Г
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          g
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:y
7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
2nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2GatherV28nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
4nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1GatherV28nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0Bnyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ќ
.nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ProdProd;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:08nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ‘
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1Prod=nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1:output:0:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concatConcatV27nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:07nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0>nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stackPack7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod:output:09nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:њ
3nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose	Transposeinputs_29nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<к
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReshapeReshape7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose:y:08nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€л
0nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMulMatMul:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Reshape:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А|
1nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аy
7nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
2nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1ConcatV2;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2:output:0@nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:и
)nyan_encoder/ecc_conv_2/FGN_out/TensordotReshape:nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMul:product:0;nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<А≥
6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpReadVariableOp?nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0б
'nyan_encoder/ecc_conv_2/FGN_out/BiasAddBiasAdd2nyan_encoder/ecc_conv_2/FGN_out/Tensordot:output:0>nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аr
'nyan_encoder/ecc_conv_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€i
'nyan_encoder/ecc_conv_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : —
%nyan_encoder/ecc_conv_2/Reshape/shapePack0nyan_encoder/ecc_conv_2/Reshape/shape/0:output:0.nyan_encoder/ecc_conv_2/strided_slice:output:0.nyan_encoder/ecc_conv_2/strided_slice:output:00nyan_encoder/ecc_conv_2/Reshape/shape/3:output:00nyan_encoder/ecc_conv_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
: 
nyan_encoder/ecc_conv_2/ReshapeReshape0nyan_encoder/ecc_conv_2/FGN_out/BiasAdd:output:0.nyan_encoder/ecc_conv_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  В
-nyan_encoder/ecc_conv_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            Д
/nyan_encoder/ecc_conv_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ъ
'nyan_encoder/ecc_conv_2/strided_slice_2StridedSlice nyan_encoder/ecc_conv_2/Cast:y:06nyan_encoder/ecc_conv_2/strided_slice_2/stack:output:08nyan_encoder/ecc_conv_2/strided_slice_2/stack_1:output:08nyan_encoder/ecc_conv_2/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskЉ
nyan_encoder/ecc_conv_2/mulMul(nyan_encoder/ecc_conv_2/Reshape:output:00nyan_encoder/ecc_conv_2/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  м
%nyan_encoder/ecc_conv_2/einsum/EinsumEinsumnyan_encoder/ecc_conv_2/mul:z:0=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdМ
nyan_encoder/ecc_conv_2/Shape_2Shape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:Г
nyan_encoder/ecc_conv_2/unstackUnpack(nyan_encoder/ecc_conv_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num¶
.nyan_encoder/ecc_conv_2/Shape_3/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0p
nyan_encoder/ecc_conv_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Г
!nyan_encoder/ecc_conv_2/unstack_1Unpack(nyan_encoder/ecc_conv_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
numx
'nyan_encoder/ecc_conv_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ѕ
!nyan_encoder/ecc_conv_2/Reshape_1Reshape=nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:00nyan_encoder/ecc_conv_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ®
0nyan_encoder/ecc_conv_2/transpose/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0w
&nyan_encoder/ecc_conv_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!nyan_encoder/ecc_conv_2/transpose	Transpose8nyan_encoder/ecc_conv_2/transpose/ReadVariableOp:value:0/nyan_encoder/ecc_conv_2/transpose/perm:output:0*
T0*
_output_shapes

:  x
'nyan_encoder/ecc_conv_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ѓ
!nyan_encoder/ecc_conv_2/Reshape_2Reshape%nyan_encoder/ecc_conv_2/transpose:y:00nyan_encoder/ecc_conv_2/Reshape_2/shape:output:0*
T0*
_output_shapes

:  ≤
nyan_encoder/ecc_conv_2/MatMulMatMul*nyan_encoder/ecc_conv_2/Reshape_1:output:0*nyan_encoder/ecc_conv_2/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ k
)nyan_encoder/ecc_conv_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<k
)nyan_encoder/ecc_conv_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
'nyan_encoder/ecc_conv_2/Reshape_3/shapePack(nyan_encoder/ecc_conv_2/unstack:output:02nyan_encoder/ecc_conv_2/Reshape_3/shape/1:output:02nyan_encoder/ecc_conv_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:Њ
!nyan_encoder/ecc_conv_2/Reshape_3Reshape(nyan_encoder/ecc_conv_2/MatMul:product:00nyan_encoder/ecc_conv_2/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ґ
nyan_encoder/ecc_conv_2/addAddV2.nyan_encoder/ecc_conv_2/einsum/Einsum:output:0*nyan_encoder/ecc_conv_2/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ґ
.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpReadVariableOp7nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
nyan_encoder/ecc_conv_2/BiasAddBiasAddnyan_encoder/ecc_conv_2/add:z:06nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< є
nyan_encoder/ecc_conv_2/mul_1Mul(nyan_encoder/ecc_conv_2/BiasAdd:output:03nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Ь
/nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu	LeakyRelu!nyan_encoder/ecc_conv_2/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=}
2nyan_encoder/global_sum_pool/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€’
 nyan_encoder/global_sum_pool/SumSum=nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu:activations:0;nyan_encoder/global_sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Я
*nyan_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp3nyan_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0Ј
nyan_encoder/dense_1/MatMulMatMul)nyan_encoder/global_sum_pool/Sum:output:02nyan_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp4nyan_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_encoder/dense_1/BiasAddBiasAdd%nyan_encoder/dense_1/MatMul:product:03nyan_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu	LeakyRelu%nyan_encoder/dense_1/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=k
nyan_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ї
nyan_encoder/flatten/ReshapeReshape:nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu:activations:0#nyan_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А†
*nyan_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp3nyan_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0≥
nyan_encoder/dense_2/MatMulMatMul%nyan_encoder/flatten/Reshape:output:02nyan_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp4nyan_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_encoder/dense_2/BiasAddBiasAdd%nyan_encoder/dense_2/MatMul:product:03nyan_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu	LeakyRelu%nyan_encoder/dense_2/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=Э
)nyan_encoder/z_mean/MatMul/ReadVariableOpReadVariableOp2nyan_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0≈
nyan_encoder/z_mean/MatMulMatMul:nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:01nyan_encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
*nyan_encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp3nyan_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0≤
nyan_encoder/z_mean/BiasAddBiasAdd$nyan_encoder/z_mean/MatMul:product:02nyan_encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@£
,nyan_encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp5nyan_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0Ћ
nyan_encoder/z_log_var/MatMulMatMul:nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:04nyan_encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@†
-nyan_encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp6nyan_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ї
nyan_encoder/z_log_var/BiasAddBiasAdd'nyan_encoder/z_log_var/MatMul:product:05nyan_encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@o
nyan_encoder/sampling/ShapeShape$nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:s
)nyan_encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+nyan_encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+nyan_encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#nyan_encoder/sampling/strided_sliceStridedSlice$nyan_encoder/sampling/Shape:output:02nyan_encoder/sampling/strided_slice/stack:output:04nyan_encoder/sampling/strided_slice/stack_1:output:04nyan_encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
nyan_encoder/sampling/Shape_1Shape$nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:u
+nyan_encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-nyan_encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-nyan_encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%nyan_encoder/sampling/strided_slice_1StridedSlice&nyan_encoder/sampling/Shape_1:output:04nyan_encoder/sampling/strided_slice_1/stack:output:06nyan_encoder/sampling/strided_slice_1/stack_1:output:06nyan_encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
)nyan_encoder/sampling/random_normal/shapePack,nyan_encoder/sampling/strided_slice:output:0.nyan_encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:m
(nyan_encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    o
*nyan_encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?в
8nyan_encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal2nyan_encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2њ ЊЎ
'nyan_encoder/sampling/random_normal/mulMulAnyan_encoder/sampling/random_normal/RandomStandardNormal:output:03nyan_encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
#nyan_encoder/sampling/random_normalAddV2+nyan_encoder/sampling/random_normal/mul:z:01nyan_encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@`
nyan_encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
nyan_encoder/sampling/mulMul$nyan_encoder/sampling/mul/x:output:0'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
nyan_encoder/sampling/ExpExpnyan_encoder/sampling/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
nyan_encoder/sampling/mul_1Mulnyan_encoder/sampling/Exp:y:0'nyan_encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ы
nyan_encoder/sampling/addAddV2$nyan_encoder/z_mean/BiasAdd:output:0nyan_encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@`
nyan_encoder/sampling/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
nyan_encoder/sampling/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    К
$nyan_encoder/sampling/ReadVariableOpReadVariableOp-nyan_encoder_sampling_readvariableop_resource*
_output_shapes
: *
dtype0b
nyan_encoder/sampling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<Щ
nyan_encoder/sampling/mul_2Mul&nyan_encoder/sampling/mul_2/x:output:0,nyan_encoder/sampling/ReadVariableOp:value:0*
T0*
_output_shapes
: `
nyan_encoder/sampling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>И
nyan_encoder/sampling/subSubnyan_encoder/sampling/mul_2:z:0$nyan_encoder/sampling/sub/y:output:0*
T0*
_output_shapes
: Р
nyan_encoder/sampling/MaximumMaximum&nyan_encoder/sampling/Const_1:output:0nyan_encoder/sampling/sub:z:0*
T0*
_output_shapes
: Т
nyan_encoder/sampling/MinimumMinimum$nyan_encoder/sampling/Const:output:0!nyan_encoder/sampling/Maximum:z:0*
T0*
_output_shapes
: b
nyan_encoder/sampling/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
nyan_encoder/sampling/add_1AddV2&nyan_encoder/sampling/add_1/x:output:0'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
nyan_encoder/sampling/SquareSquare$nyan_encoder/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
nyan_encoder/sampling/sub_1Subnyan_encoder/sampling/add_1:z:0 nyan_encoder/sampling/Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@}
nyan_encoder/sampling/Exp_1Exp'nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
nyan_encoder/sampling/sub_2Subnyan_encoder/sampling/sub_1:z:0nyan_encoder/sampling/Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@b
nyan_encoder/sampling/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њЭ
nyan_encoder/sampling/mul_3Mul&nyan_encoder/sampling/mul_3/x:output:0nyan_encoder/sampling/sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@m
+nyan_encoder/sampling/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :•
nyan_encoder/sampling/SumSumnyan_encoder/sampling/mul_3:z:04nyan_encoder/sampling/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
nyan_encoder/sampling/Const_2Const*
_output_shapes
:*
dtype0*
valueB: П
nyan_encoder/sampling/MeanMean"nyan_encoder/sampling/Sum:output:0&nyan_encoder/sampling/Const_2:output:0*
T0*
_output_shapes
: Л
nyan_encoder/sampling/mul_4Mul!nyan_encoder/sampling/Minimum:z:0#nyan_encoder/sampling/Mean:output:0*
T0*
_output_shapes
: d
nyan_encoder/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFФ
nyan_encoder/sampling/truedivRealDivnyan_encoder/sampling/mul_4:z:0(nyan_encoder/sampling/truediv/y:output:0*
T0*
_output_shapes
: \
nyan_encoder/sampling/RankConst*
_output_shapes
: *
dtype0*
value	B : c
!nyan_encoder/sampling/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!nyan_encoder/sampling/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ї
nyan_encoder/sampling/rangeRange*nyan_encoder/sampling/range/start:output:0#nyan_encoder/sampling/Rank:output:0*nyan_encoder/sampling/range/delta:output:0*
_output_shapes
: О
nyan_encoder/sampling/Sum_1Sum#nyan_encoder/sampling/Mean:output:0$nyan_encoder/sampling/range:output:0*
T0*
_output_shapes
: љ
)nyan_encoder/sampling/AssignAddVariableOpAssignAddVariableOp2nyan_encoder_sampling_assignaddvariableop_resource$nyan_encoder/sampling/Sum_1:output:0*
_output_shapes
 *
dtype0\
nyan_encoder/sampling/SizeConst*
_output_shapes
: *
dtype0*
value	B :w
nyan_encoder/sampling/CastCast#nyan_encoder/sampling/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: з
+nyan_encoder/sampling/AssignAddVariableOp_1AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_1_resourcenyan_encoder/sampling/Cast:y:0*^nyan_encoder/sampling/AssignAddVariableOp*
_output_shapes
 *
dtype0ф
/nyan_encoder/sampling/div_no_nan/ReadVariableOpReadVariableOp2nyan_encoder_sampling_assignaddvariableop_resource*^nyan_encoder/sampling/AssignAddVariableOp,^nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0ћ
1nyan_encoder/sampling/div_no_nan/ReadVariableOp_1ReadVariableOp4nyan_encoder_sampling_assignaddvariableop_1_resource,^nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ѕ
 nyan_encoder/sampling/div_no_nanDivNoNan7nyan_encoder/sampling/div_no_nan/ReadVariableOp:value:09nyan_encoder/sampling/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: q
nyan_encoder/sampling/IdentityIdentity$nyan_encoder/sampling/div_no_nan:z:0*
T0*
_output_shapes
: ^
nyan_encoder/sampling/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : e
#nyan_encoder/sampling/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : e
#nyan_encoder/sampling/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :√
nyan_encoder/sampling/range_1Range,nyan_encoder/sampling/range_1/start:output:0%nyan_encoder/sampling/Rank_1:output:0,nyan_encoder/sampling/range_1/delta:output:0*
_output_shapes
: О
nyan_encoder/sampling/Sum_2Sum!nyan_encoder/sampling/truediv:z:0&nyan_encoder/sampling/range_1:output:0*
T0*
_output_shapes
: Ѕ
+nyan_encoder/sampling/AssignAddVariableOp_2AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_2_resource$nyan_encoder/sampling/Sum_2:output:0*
_output_shapes
 *
dtype0^
nyan_encoder/sampling/Size_1Const*
_output_shapes
: *
dtype0*
value	B :{
nyan_encoder/sampling/Cast_1Cast%nyan_encoder/sampling/Size_1:output:0*

DstT0*

SrcT0*
_output_shapes
: л
+nyan_encoder/sampling/AssignAddVariableOp_3AssignAddVariableOp4nyan_encoder_sampling_assignaddvariableop_3_resource nyan_encoder/sampling/Cast_1:y:0,^nyan_encoder/sampling/AssignAddVariableOp_2*
_output_shapes
 *
dtype0ъ
1nyan_encoder/sampling/div_no_nan_1/ReadVariableOpReadVariableOp4nyan_encoder_sampling_assignaddvariableop_2_resource,^nyan_encoder/sampling/AssignAddVariableOp_2,^nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0ќ
3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1ReadVariableOp4nyan_encoder_sampling_assignaddvariableop_3_resource,^nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0«
"nyan_encoder/sampling/div_no_nan_1DivNoNan9nyan_encoder/sampling/div_no_nan_1/ReadVariableOp:value:0;nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: u
 nyan_encoder/sampling/Identity_1Identity&nyan_encoder/sampling/div_no_nan_1:z:0*
T0*
_output_shapes
: Я
*nyan_decoder/dense_3/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Ђ
nyan_decoder/dense_3/MatMulMatMulnyan_encoder/sampling/add:z:02nyan_decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
+nyan_decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
nyan_decoder/dense_3/BiasAddBiasAdd%nyan_decoder/dense_3/MatMul:product:03nyan_decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЪ
,nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu	LeakyRelu%nyan_decoder/dense_3/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=†
*nyan_decoder/dense_4/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0»
nyan_decoder/dense_4/MatMulMatMul:nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:02nyan_decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІЭ
+nyan_decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0ґ
nyan_decoder/dense_4/BiasAddBiasAdd%nyan_decoder/dense_4/MatMul:product:03nyan_decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІБ
nyan_decoder/dense_4/SigmoidSigmoid%nyan_decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€І†
*nyan_decoder/dense_5/MatMul/ReadVariableOpReadVariableOp3nyan_decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0»
nyan_decoder/dense_5/MatMulMatMul:nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:02nyan_decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ЌЭ
+nyan_decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp4nyan_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0ґ
nyan_decoder/dense_5/BiasAddBiasAdd%nyan_decoder/dense_5/MatMul:product:03nyan_decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќc
nyan_decoder/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€«
nyan_decoder/concatConcatV2 nyan_decoder/dense_4/Sigmoid:y:0%nyan_decoder/dense_5/BiasAdd:output:0!nyan_decoder/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€фl
IdentityIdentitynyan_decoder/concat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фa

Identity_1Identity!nyan_encoder/sampling/truediv:z:0^NoOp*
T0*
_output_shapes
: ћ
NoOpNoOp,^nyan_decoder/dense_3/BiasAdd/ReadVariableOp+^nyan_decoder/dense_3/MatMul/ReadVariableOp,^nyan_decoder/dense_4/BiasAdd/ReadVariableOp+^nyan_decoder/dense_4/MatMul/ReadVariableOp,^nyan_decoder/dense_5/BiasAdd/ReadVariableOp+^nyan_decoder/dense_5/MatMul/ReadVariableOp*^nyan_encoder/dense/BiasAdd/ReadVariableOp,^nyan_encoder/dense/Tensordot/ReadVariableOp,^nyan_encoder/dense_1/BiasAdd/ReadVariableOp+^nyan_encoder/dense_1/MatMul/ReadVariableOp,^nyan_encoder/dense_2/BiasAdd/ReadVariableOp+^nyan_encoder/dense_2/MatMul/ReadVariableOp-^nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp5^nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp/^nyan_encoder/ecc_conv/transpose/ReadVariableOp/^nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp9^nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp1^nyan_encoder/ecc_conv_1/transpose/ReadVariableOp/^nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp7^nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp9^nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp1^nyan_encoder/ecc_conv_2/transpose/ReadVariableOp*^nyan_encoder/sampling/AssignAddVariableOp,^nyan_encoder/sampling/AssignAddVariableOp_1,^nyan_encoder/sampling/AssignAddVariableOp_2,^nyan_encoder/sampling/AssignAddVariableOp_3%^nyan_encoder/sampling/ReadVariableOp0^nyan_encoder/sampling/div_no_nan/ReadVariableOp2^nyan_encoder/sampling/div_no_nan/ReadVariableOp_12^nyan_encoder/sampling/div_no_nan_1/ReadVariableOp4^nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1.^nyan_encoder/z_log_var/BiasAdd/ReadVariableOp-^nyan_encoder/z_log_var/MatMul/ReadVariableOp+^nyan_encoder/z_mean/BiasAdd/ReadVariableOp*^nyan_encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+nyan_decoder/dense_3/BiasAdd/ReadVariableOp+nyan_decoder/dense_3/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_3/MatMul/ReadVariableOp*nyan_decoder/dense_3/MatMul/ReadVariableOp2Z
+nyan_decoder/dense_4/BiasAdd/ReadVariableOp+nyan_decoder/dense_4/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_4/MatMul/ReadVariableOp*nyan_decoder/dense_4/MatMul/ReadVariableOp2Z
+nyan_decoder/dense_5/BiasAdd/ReadVariableOp+nyan_decoder/dense_5/BiasAdd/ReadVariableOp2X
*nyan_decoder/dense_5/MatMul/ReadVariableOp*nyan_decoder/dense_5/MatMul/ReadVariableOp2V
)nyan_encoder/dense/BiasAdd/ReadVariableOp)nyan_encoder/dense/BiasAdd/ReadVariableOp2Z
+nyan_encoder/dense/Tensordot/ReadVariableOp+nyan_encoder/dense/Tensordot/ReadVariableOp2Z
+nyan_encoder/dense_1/BiasAdd/ReadVariableOp+nyan_encoder/dense_1/BiasAdd/ReadVariableOp2X
*nyan_encoder/dense_1/MatMul/ReadVariableOp*nyan_encoder/dense_1/MatMul/ReadVariableOp2Z
+nyan_encoder/dense_2/BiasAdd/ReadVariableOp+nyan_encoder/dense_2/BiasAdd/ReadVariableOp2X
*nyan_encoder/dense_2/MatMul/ReadVariableOp*nyan_encoder/dense_2/MatMul/ReadVariableOp2\
,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp,nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp2l
4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp4nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp6nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp2`
.nyan_encoder/ecc_conv/transpose/ReadVariableOp.nyan_encoder/ecc_conv/transpose/ReadVariableOp2`
.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp.nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp6nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2t
8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp8nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2d
0nyan_encoder/ecc_conv_1/transpose/ReadVariableOp0nyan_encoder/ecc_conv_1/transpose/ReadVariableOp2`
.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp.nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp2p
6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp6nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp2t
8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp8nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp2d
0nyan_encoder/ecc_conv_2/transpose/ReadVariableOp0nyan_encoder/ecc_conv_2/transpose/ReadVariableOp2V
)nyan_encoder/sampling/AssignAddVariableOp)nyan_encoder/sampling/AssignAddVariableOp2Z
+nyan_encoder/sampling/AssignAddVariableOp_1+nyan_encoder/sampling/AssignAddVariableOp_12Z
+nyan_encoder/sampling/AssignAddVariableOp_2+nyan_encoder/sampling/AssignAddVariableOp_22Z
+nyan_encoder/sampling/AssignAddVariableOp_3+nyan_encoder/sampling/AssignAddVariableOp_32L
$nyan_encoder/sampling/ReadVariableOp$nyan_encoder/sampling/ReadVariableOp2b
/nyan_encoder/sampling/div_no_nan/ReadVariableOp/nyan_encoder/sampling/div_no_nan/ReadVariableOp2f
1nyan_encoder/sampling/div_no_nan/ReadVariableOp_11nyan_encoder/sampling/div_no_nan/ReadVariableOp_12f
1nyan_encoder/sampling/div_no_nan_1/ReadVariableOp1nyan_encoder/sampling/div_no_nan_1/ReadVariableOp2j
3nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_13nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_12^
-nyan_encoder/z_log_var/BiasAdd/ReadVariableOp-nyan_encoder/z_log_var/BiasAdd/ReadVariableOp2\
,nyan_encoder/z_log_var/MatMul/ReadVariableOp,nyan_encoder/z_log_var/MatMul/ReadVariableOp2X
*nyan_encoder/z_mean/BiasAdd/ReadVariableOp*nyan_encoder/z_mean/BiasAdd/ReadVariableOp2V
)nyan_encoder/z_mean/MatMul/ReadVariableOp)nyan_encoder/z_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
ф
€
%__inference_signature_wrapper_7409409
input_1
input_2	
input_3
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26:	@А

unknown_27:	А

unknown_28:
АІ

unknown_29:	І

unknown_30:
АЌ

unknown_31:	Ќ
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_7407447p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
ѕ
щ
B__inference_dense_layer_call_and_return_conditional_losses_7411298

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<q
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=v
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
Є
`
D__inference_flatten_layer_call_and_return_conditional_losses_7411631

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
’

ч
D__inference_dense_3_layer_call_and_return_conditional_losses_7411805

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_6/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ё
∆
.__inference_nyan_encoder_layer_call_fn_7408429
x
a	
e
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallxaeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:€€€€€€€€€<

_user_specified_namex:NJ
+
_output_shapes
:€€€€€€€€€<<

_user_specified_namea:RN
/
_output_shapes
:€€€€€€€€€<<

_user_specified_namee
пE
÷

I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7407957

inputs
inputs_1	
inputs_2
dense_7407504:
dense_7407506:#
ecc_conv_7407591:	А
ecc_conv_7407593:	А"
ecc_conv_7407595: 
ecc_conv_7407597: %
ecc_conv_1_7407682:	А!
ecc_conv_1_7407684:	А$
ecc_conv_1_7407686:   
ecc_conv_1_7407688: %
ecc_conv_2_7407773:	А!
ecc_conv_2_7407775:	А$
ecc_conv_2_7407777:   
ecc_conv_2_7407779: "
dense_1_7407802:	 А
dense_1_7407804:	А#
dense_2_7407827:
АА
dense_2_7407829:	А!
z_mean_7407843:	А@
z_mean_7407845:@$
z_log_var_7407859:	А@
z_log_var_7407861:@
sampling_7407943: 
sampling_7407945: 
sampling_7407947: 
sampling_7407949: 
sampling_7407951: 
identity

identity_1ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ ecc_conv/StatefulPartitionedCallҐ"ecc_conv_1/StatefulPartitionedCallҐ"ecc_conv_2/StatefulPartitionedCallҐ sampling/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall 
graph_masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7407466r
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      £
graph_masking/strided_sliceStridedSliceinputs*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskО
dense/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0dense_7407504dense_7407506*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7407503€
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_7407591ecc_conv_7407593ecc_conv_7407595ecc_conv_7407597*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590О
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_1_7407682ecc_conv_1_7407684ecc_conv_1_7407686ecc_conv_1_7407688*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681Р
"ecc_conv_2/StatefulPartitionedCallStatefulPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_2_7407773ecc_conv_2_7407775ecc_conv_2_7407777ecc_conv_2_7407779*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772п
global_sum_pool/PartitionedCallPartitionedCall+ecc_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788Х
dense_1/StatefulPartitionedCallStatefulPartitionedCall(global_sum_pool/PartitionedCall:output:0dense_1_7407802dense_1_7407804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801Ё
flatten/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7407813Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_7407827dense_2_7407829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826Р
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_7407843z_mean_7407845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842Ь
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_7407859z_log_var_7407861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858€
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0sampling_7407943sampling_7407945sampling_7407947sampling_7407949sampling_7407951*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_7407942x
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@i

Identity_1Identity)sampling/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: €
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall#^ecc_conv_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall2H
"ecc_conv_2/StatefulPartitionedCall"ecc_conv_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs
Љ
K
/__inference_graph_masking_layer_call_fn_7411242

inputs
identityЉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7408162d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
ф
µ
@__inference_vae_layer_call_and_return_conditional_losses_7409254
input_1
input_2	
input_3&
nyan_encoder_7409183:"
nyan_encoder_7409185:'
nyan_encoder_7409187:	А#
nyan_encoder_7409189:	А&
nyan_encoder_7409191: "
nyan_encoder_7409193: '
nyan_encoder_7409195:	А#
nyan_encoder_7409197:	А&
nyan_encoder_7409199:  "
nyan_encoder_7409201: '
nyan_encoder_7409203:	А#
nyan_encoder_7409205:	А&
nyan_encoder_7409207:  "
nyan_encoder_7409209: '
nyan_encoder_7409211:	 А#
nyan_encoder_7409213:	А(
nyan_encoder_7409215:
АА#
nyan_encoder_7409217:	А'
nyan_encoder_7409219:	А@"
nyan_encoder_7409221:@'
nyan_encoder_7409223:	А@"
nyan_encoder_7409225:@
nyan_encoder_7409227: 
nyan_encoder_7409229: 
nyan_encoder_7409231: 
nyan_encoder_7409233: 
nyan_encoder_7409235: '
nyan_decoder_7409239:	@А#
nyan_decoder_7409241:	А(
nyan_decoder_7409243:
АІ#
nyan_decoder_7409245:	І(
nyan_decoder_7409247:
АЌ#
nyan_decoder_7409249:	Ќ
identity

identity_1ИҐ$nyan_decoder/StatefulPartitionedCallҐ$nyan_encoder/StatefulPartitionedCallт
$nyan_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3nyan_encoder_7409183nyan_encoder_7409185nyan_encoder_7409187nyan_encoder_7409189nyan_encoder_7409191nyan_encoder_7409193nyan_encoder_7409195nyan_encoder_7409197nyan_encoder_7409199nyan_encoder_7409201nyan_encoder_7409203nyan_encoder_7409205nyan_encoder_7409207nyan_encoder_7409209nyan_encoder_7409211nyan_encoder_7409213nyan_encoder_7409215nyan_encoder_7409217nyan_encoder_7409219nyan_encoder_7409221nyan_encoder_7409223nyan_encoder_7409225nyan_encoder_7409227nyan_encoder_7409229nyan_encoder_7409231nyan_encoder_7409233nyan_encoder_7409235*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7407957О
$nyan_decoder/StatefulPartitionedCallStatefulPartitionedCall-nyan_encoder/StatefulPartitionedCall:output:0nyan_decoder_7409239nyan_decoder_7409241nyan_decoder_7409243nyan_decoder_7409245nyan_decoder_7409247nyan_decoder_7409249*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645}
IdentityIdentity-nyan_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фm

Identity_1Identity-nyan_encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ф
NoOpNoOp%^nyan_decoder/StatefulPartitionedCall%^nyan_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$nyan_decoder/StatefulPartitionedCall$nyan_decoder/StatefulPartitionedCall2L
$nyan_encoder/StatefulPartitionedCall$nyan_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
Ќ	
ш
F__inference_z_log_var_layer_call_and_return_conditional_losses_7411689

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
„L
а
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7411589
inputs_0
inputs_1	
inputs_2

mask_0<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<=
ShapeShapeinputs_0*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  З
einsum/EinsumEinsummul:z:0inputs_0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd?
Shape_2Shapeinputs_0*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    j
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< \
mul_1MulBiasAdd:output:0mask_0*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_3/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:€€€€€€€€€< 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
ЯE
•
E__inference_sampling_layer_call_and_return_conditional_losses_7407942

inputs
inputs_1
inputs_2: &
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 

identity_2

identity_3ИҐAssignAddVariableOpҐAssignAddVariableOp_1ҐAssignAddVariableOp_2ҐAssignAddVariableOp_3ҐReadVariableOpҐdiv_no_nan/ReadVariableOpҐdiv_no_nan/ReadVariableOp_1Ґdiv_no_nan_1/ReadVariableOpҐdiv_no_nan_1/ReadVariableOp_1;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2мБ•Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€@E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    O
ReadVariableOpReadVariableOpinputs_2*
_output_shapes
: *
dtype0L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<W
mul_2Mulmul_2/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>F
subSub	mul_2:z:0sub/y:output:0*
T0*
_output_shapes
: N
MaximumMaximumConst_1:output:0sub:z:0*
T0*
_output_shapes
: P
MinimumMinimumConst:output:0Maximum:z:0*
T0*
_output_shapes
: L
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?\
add_1AddV2add_1/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€@J
SquareSquareinputs*
T0*'
_output_shapes
:€€€€€€€€€@U
sub_1Sub	add_1:z:0
Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@H
Exp_1Expinputs_1*
T0*'
_output_shapes
:€€€€€€€€€@T
sub_2Sub	sub_1:z:0	Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њ[
mul_3Mulmul_3/x:output:0	sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :c
SumSum	mul_3:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: M
MeanMeanSum:output:0Const_2:output:0*
T0*
_output_shapes
: I
mul_4MulMinimum:z:0Mean:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jFR
truedivRealDiv	mul_4:z:0truediv/y:output:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: L
Sum_1SumMean:output:0range:output:0*
T0*
_output_shapes
: {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum_1:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: П
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0Ь
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0К
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: L
Sum_2Sumtruediv:z:0range_1:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*
_output_shapes
 *
dtype0H
Size_1Const*
_output_shapes
: *
dtype0*
value	B :O
Cast_1CastSize_1:output:0*

DstT0*

SrcT0*
_output_shapes
: У
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_1:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0Ґ
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2^AssignAddVariableOp_3*
_output_shapes
: *
dtype0М
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0Е
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: X

Identity_2Identityadd:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@K

Identity_3Identitytruediv:z:0^NoOp*
T0*
_output_shapes
: ≠
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€@:€€€€€€€€€@: : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32 
ReadVariableOpReadVariableOp26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_1:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
¶

ш
D__inference_dense_4_layer_call_and_return_conditional_losses_7411825

inputs2
matmul_readvariableop_resource:
АІ.
biasadd_readvariableop_resource:	І
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Іs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€І[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Іw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
 
щ
*__inference_ecc_conv_layer_call_fn_7411314
inputs_0
inputs_1	
inputs_2

mask_0
unknown:	А
	unknown_0:	А
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2mask_0unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2:SO
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_namemask/0
Гµ
є#
"__inference__wrapped_model_7407447
input_1
input_2	
input_3J
8vae_nyan_encoder_dense_tensordot_readvariableop_resource:D
6vae_nyan_encoder_dense_biasadd_readvariableop_resource:V
Cvae_nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource:	АP
Avae_nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource:	АK
9vae_nyan_encoder_ecc_conv_shape_3_readvariableop_resource: G
9vae_nyan_encoder_ecc_conv_biasadd_readvariableop_resource: X
Evae_nyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	АR
Cvae_nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	АM
;vae_nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource:  I
;vae_nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource: X
Evae_nyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource:	АR
Cvae_nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource:	АM
;vae_nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource:  I
;vae_nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource: J
7vae_nyan_encoder_dense_1_matmul_readvariableop_resource:	 АG
8vae_nyan_encoder_dense_1_biasadd_readvariableop_resource:	АK
7vae_nyan_encoder_dense_2_matmul_readvariableop_resource:
ААG
8vae_nyan_encoder_dense_2_biasadd_readvariableop_resource:	АI
6vae_nyan_encoder_z_mean_matmul_readvariableop_resource:	А@E
7vae_nyan_encoder_z_mean_biasadd_readvariableop_resource:@L
9vae_nyan_encoder_z_log_var_matmul_readvariableop_resource:	А@H
:vae_nyan_encoder_z_log_var_biasadd_readvariableop_resource:@;
1vae_nyan_encoder_sampling_readvariableop_resource: @
6vae_nyan_encoder_sampling_assignaddvariableop_resource: B
8vae_nyan_encoder_sampling_assignaddvariableop_1_resource: B
8vae_nyan_encoder_sampling_assignaddvariableop_2_resource: B
8vae_nyan_encoder_sampling_assignaddvariableop_3_resource: J
7vae_nyan_decoder_dense_3_matmul_readvariableop_resource:	@АG
8vae_nyan_decoder_dense_3_biasadd_readvariableop_resource:	АK
7vae_nyan_decoder_dense_4_matmul_readvariableop_resource:
АІG
8vae_nyan_decoder_dense_4_biasadd_readvariableop_resource:	ІK
7vae_nyan_decoder_dense_5_matmul_readvariableop_resource:
АЌG
8vae_nyan_decoder_dense_5_biasadd_readvariableop_resource:	Ќ
identityИҐ/vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOpҐ.vae/nyan_decoder/dense_3/MatMul/ReadVariableOpҐ/vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOpҐ.vae/nyan_decoder/dense_4/MatMul/ReadVariableOpҐ/vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOpҐ.vae/nyan_decoder/dense_5/MatMul/ReadVariableOpҐ-vae/nyan_encoder/dense/BiasAdd/ReadVariableOpҐ/vae/nyan_encoder/dense/Tensordot/ReadVariableOpҐ/vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOpҐ.vae/nyan_encoder/dense_1/MatMul/ReadVariableOpҐ/vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOpҐ.vae/nyan_encoder/dense_2/MatMul/ReadVariableOpҐ0vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpҐ8vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpҐ:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpҐ2vae/nyan_encoder/ecc_conv/transpose/ReadVariableOpҐ2vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpҐ:vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpҐ<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpҐ4vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOpҐ2vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpҐ:vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpҐ<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpҐ4vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOpҐ-vae/nyan_encoder/sampling/AssignAddVariableOpҐ/vae/nyan_encoder/sampling/AssignAddVariableOp_1Ґ/vae/nyan_encoder/sampling/AssignAddVariableOp_2Ґ/vae/nyan_encoder/sampling/AssignAddVariableOp_3Ґ(vae/nyan_encoder/sampling/ReadVariableOpҐ3vae/nyan_encoder/sampling/div_no_nan/ReadVariableOpҐ5vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_1Ґ5vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOpҐ7vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1Ґ1vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOpҐ0vae/nyan_encoder/z_log_var/MatMul/ReadVariableOpҐ.vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOpҐ-vae/nyan_encoder/z_mean/MatMul/ReadVariableOpГ
2vae/nyan_encoder/graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4vae/nyan_encoder/graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€Е
4vae/nyan_encoder/graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      к
,vae/nyan_encoder/graph_masking/strided_sliceStridedSliceinput_1;vae/nyan_encoder/graph_masking/strided_slice/stack:output:0=vae/nyan_encoder/graph_masking/strided_slice/stack_1:output:0=vae/nyan_encoder/graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskЕ
4vae/nyan_encoder/graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€З
6vae/nyan_encoder/graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        З
6vae/nyan_encoder/graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      р
.vae/nyan_encoder/graph_masking/strided_slice_1StridedSliceinput_1=vae/nyan_encoder/graph_masking/strided_slice_1/stack:output:0?vae/nyan_encoder/graph_masking/strided_slice_1/stack_1:output:0?vae/nyan_encoder/graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_mask®
/vae/nyan_encoder/dense/Tensordot/ReadVariableOpReadVariableOp8vae_nyan_encoder_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0o
%vae/nyan_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%vae/nyan_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Л
&vae/nyan_encoder/dense/Tensordot/ShapeShape5vae/nyan_encoder/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:p
.vae/nyan_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
)vae/nyan_encoder/dense/Tensordot/GatherV2GatherV2/vae/nyan_encoder/dense/Tensordot/Shape:output:0.vae/nyan_encoder/dense/Tensordot/free:output:07vae/nyan_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0vae/nyan_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
+vae/nyan_encoder/dense/Tensordot/GatherV2_1GatherV2/vae/nyan_encoder/dense/Tensordot/Shape:output:0.vae/nyan_encoder/dense/Tensordot/axes:output:09vae/nyan_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&vae/nyan_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ≥
%vae/nyan_encoder/dense/Tensordot/ProdProd2vae/nyan_encoder/dense/Tensordot/GatherV2:output:0/vae/nyan_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(vae/nyan_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: є
'vae/nyan_encoder/dense/Tensordot/Prod_1Prod4vae/nyan_encoder/dense/Tensordot/GatherV2_1:output:01vae/nyan_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,vae/nyan_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
'vae/nyan_encoder/dense/Tensordot/concatConcatV2.vae/nyan_encoder/dense/Tensordot/free:output:0.vae/nyan_encoder/dense/Tensordot/axes:output:05vae/nyan_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Њ
&vae/nyan_encoder/dense/Tensordot/stackPack.vae/nyan_encoder/dense/Tensordot/Prod:output:00vae/nyan_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:÷
*vae/nyan_encoder/dense/Tensordot/transpose	Transpose5vae/nyan_encoder/graph_masking/strided_slice:output:00vae/nyan_encoder/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<ѕ
(vae/nyan_encoder/dense/Tensordot/ReshapeReshape.vae/nyan_encoder/dense/Tensordot/transpose:y:0/vae/nyan_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ѕ
'vae/nyan_encoder/dense/Tensordot/MatMulMatMul1vae/nyan_encoder/dense/Tensordot/Reshape:output:07vae/nyan_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
(vae/nyan_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:p
.vae/nyan_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
)vae/nyan_encoder/dense/Tensordot/concat_1ConcatV22vae/nyan_encoder/dense/Tensordot/GatherV2:output:01vae/nyan_encoder/dense/Tensordot/Const_2:output:07vae/nyan_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:»
 vae/nyan_encoder/dense/TensordotReshape1vae/nyan_encoder/dense/Tensordot/MatMul:product:02vae/nyan_encoder/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<†
-vae/nyan_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp6vae_nyan_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѕ
vae/nyan_encoder/dense/BiasAddBiasAdd)vae/nyan_encoder/dense/Tensordot:output:05vae/nyan_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<Я
,vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu	LeakyRelu'vae/nyan_encoder/dense/BiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=t
vae/nyan_encoder/ecc_conv/CastCastinput_2*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<Й
vae/nyan_encoder/ecc_conv/ShapeShape:vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:А
-vae/nyan_encoder/ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€В
/vae/nyan_encoder/ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€y
/vae/nyan_encoder/ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'vae/nyan_encoder/ecc_conv/strided_sliceStridedSlice(vae/nyan_encoder/ecc_conv/Shape:output:06vae/nyan_encoder/ecc_conv/strided_slice/stack:output:08vae/nyan_encoder/ecc_conv/strided_slice/stack_1:output:08vae/nyan_encoder/ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
!vae/nyan_encoder/ecc_conv/Shape_1Shape:vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:В
/vae/nyan_encoder/ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1vae/nyan_encoder/ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1vae/nyan_encoder/ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)vae/nyan_encoder/ecc_conv/strided_slice_1StridedSlice*vae/nyan_encoder/ecc_conv/Shape_1:output:08vae/nyan_encoder/ecc_conv/strided_slice_1/stack:output:0:vae/nyan_encoder/ecc_conv/strided_slice_1/stack_1:output:0:vae/nyan_encoder/ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOpCvae_nyan_encoder_ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0z
0vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Е
0vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          h
1vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ShapeShapeinput_3*
T0*
_output_shapes
:{
9vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
4vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2GatherV2:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:09vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:0Bvae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
6vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Shape:output:09vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0Dvae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ‘
0vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ProdProd=vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:0:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Џ
2vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1Prod?vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0<vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : §
2vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concatConcatV29vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/free:output:09vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/axes:output:0@vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:я
1vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/stackPack9vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod:output:0;vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
5vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose	Transposeinput_3;vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<р
3vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReshapeReshape9vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/transpose:y:0:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€с
2vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMulMatMul<vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Reshape:output:0Bvae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А~
3vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А{
9vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
4vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1ConcatV2=vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/GatherV2:output:0<vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/Const_2:output:0Bvae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:о
+vae/nyan_encoder/ecc_conv/FGN_out/TensordotReshape<vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/MatMul:product:0=vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АЈ
8vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOpAvae_nyan_encoder_ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0з
)vae/nyan_encoder/ecc_conv/FGN_out/BiasAddBiasAdd4vae/nyan_encoder/ecc_conv/FGN_out/Tensordot:output:0@vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аt
)vae/nyan_encoder/ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€k
)vae/nyan_encoder/ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ё
'vae/nyan_encoder/ecc_conv/Reshape/shapePack2vae/nyan_encoder/ecc_conv/Reshape/shape/0:output:00vae/nyan_encoder/ecc_conv/strided_slice:output:00vae/nyan_encoder/ecc_conv/strided_slice:output:02vae/nyan_encoder/ecc_conv/Reshape/shape/3:output:02vae/nyan_encoder/ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:–
!vae/nyan_encoder/ecc_conv/ReshapeReshape2vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd:output:00vae/nyan_encoder/ecc_conv/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< Д
/vae/nyan_encoder/ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Ж
1vae/nyan_encoder/ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            Ж
1vae/nyan_encoder/ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
)vae/nyan_encoder/ecc_conv/strided_slice_2StridedSlice"vae/nyan_encoder/ecc_conv/Cast:y:08vae/nyan_encoder/ecc_conv/strided_slice_2/stack:output:0:vae/nyan_encoder/ecc_conv/strided_slice_2/stack_1:output:0:vae/nyan_encoder/ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_mask¬
vae/nyan_encoder/ecc_conv/mulMul*vae/nyan_encoder/ecc_conv/Reshape:output:02vae/nyan_encoder/ecc_conv/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<< н
'vae/nyan_encoder/ecc_conv/einsum/EinsumEinsum!vae/nyan_encoder/ecc_conv/mul:z:0:vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdЛ
!vae/nyan_encoder/ecc_conv/Shape_2Shape:vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:З
!vae/nyan_encoder/ecc_conv/unstackUnpack*vae/nyan_encoder/ecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num™
0vae/nyan_encoder/ecc_conv/Shape_3/ReadVariableOpReadVariableOp9vae_nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0r
!vae/nyan_encoder/ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       З
#vae/nyan_encoder/ecc_conv/unstack_1Unpack*vae/nyan_encoder/ecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
numz
)vae/nyan_encoder/ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   –
#vae/nyan_encoder/ecc_conv/Reshape_1Reshape:vae/nyan_encoder/dense/leaky_re_lu/LeakyRelu:activations:02vae/nyan_encoder/ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
2vae/nyan_encoder/ecc_conv/transpose/ReadVariableOpReadVariableOp9vae_nyan_encoder_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0y
(vae/nyan_encoder/ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       »
#vae/nyan_encoder/ecc_conv/transpose	Transpose:vae/nyan_encoder/ecc_conv/transpose/ReadVariableOp:value:01vae/nyan_encoder/ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: z
)vae/nyan_encoder/ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   €€€€і
#vae/nyan_encoder/ecc_conv/Reshape_2Reshape'vae/nyan_encoder/ecc_conv/transpose:y:02vae/nyan_encoder/ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: Є
 vae/nyan_encoder/ecc_conv/MatMulMatMul,vae/nyan_encoder/ecc_conv/Reshape_1:output:0,vae/nyan_encoder/ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ m
+vae/nyan_encoder/ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<m
+vae/nyan_encoder/ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : ч
)vae/nyan_encoder/ecc_conv/Reshape_3/shapePack*vae/nyan_encoder/ecc_conv/unstack:output:04vae/nyan_encoder/ecc_conv/Reshape_3/shape/1:output:04vae/nyan_encoder/ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:ƒ
#vae/nyan_encoder/ecc_conv/Reshape_3Reshape*vae/nyan_encoder/ecc_conv/MatMul:product:02vae/nyan_encoder/ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< Љ
vae/nyan_encoder/ecc_conv/addAddV20vae/nyan_encoder/ecc_conv/einsum/Einsum:output:0,vae/nyan_encoder/ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ¶
0vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOpReadVariableOp9vae_nyan_encoder_ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0њ
!vae/nyan_encoder/ecc_conv/BiasAddBiasAdd!vae/nyan_encoder/ecc_conv/add:z:08vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Ѕ
vae/nyan_encoder/ecc_conv/mul_1Mul*vae/nyan_encoder/ecc_conv/BiasAdd:output:07vae/nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< †
1vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu	LeakyRelu#vae/nyan_encoder/ecc_conv/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=v
 vae/nyan_encoder/ecc_conv_1/CastCastinput_2*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<Р
!vae/nyan_encoder/ecc_conv_1/ShapeShape?vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:В
/vae/nyan_encoder/ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Д
1vae/nyan_encoder/ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1vae/nyan_encoder/ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)vae/nyan_encoder/ecc_conv_1/strided_sliceStridedSlice*vae/nyan_encoder/ecc_conv_1/Shape:output:08vae/nyan_encoder/ecc_conv_1/strided_slice/stack:output:0:vae/nyan_encoder/ecc_conv_1/strided_slice/stack_1:output:0:vae/nyan_encoder/ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
#vae/nyan_encoder/ecc_conv_1/Shape_1Shape?vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:Д
1vae/nyan_encoder/ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3vae/nyan_encoder/ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3vae/nyan_encoder/ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+vae/nyan_encoder/ecc_conv_1/strided_slice_1StridedSlice,vae/nyan_encoder/ecc_conv_1/Shape_1:output:0:vae/nyan_encoder/ecc_conv_1/strided_slice_1/stack:output:0<vae/nyan_encoder/ecc_conv_1/strided_slice_1/stack_1:output:0<vae/nyan_encoder/ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask√
<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOpEvae_nyan_encoder_ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0|
2vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:З
2vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          j
3vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ShapeShapeinput_3*
T0*
_output_shapes
:}
;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ћ
6vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:0;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:0Dvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѕ
8vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Shape:output:0;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0Fvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Џ
2vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ProdProd?vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: а
4vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1ProdAvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0>vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
4vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concatConcatV2;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/free:output:0;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/axes:output:0Bvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:е
3vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stackPack;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod:output:0=vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:∆
7vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose	Transposeinput_3=vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<ц
5vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReshapeReshape;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/transpose:y:0<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ч
4vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMulMatMul>vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Reshape:output:0Dvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
5vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А}
;vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
6vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2?vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0>vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/Const_2:output:0Dvae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ф
-vae/nyan_encoder/ecc_conv_1/FGN_out/TensordotReshape>vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/MatMul:product:0?vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аї
:vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOpCvae_nyan_encoder_ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0н
+vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAddBiasAdd6vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot:output:0Bvae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аv
+vae/nyan_encoder/ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€m
+vae/nyan_encoder/ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : й
)vae/nyan_encoder/ecc_conv_1/Reshape/shapePack4vae/nyan_encoder/ecc_conv_1/Reshape/shape/0:output:02vae/nyan_encoder/ecc_conv_1/strided_slice:output:02vae/nyan_encoder/ecc_conv_1/strided_slice:output:04vae/nyan_encoder/ecc_conv_1/Reshape/shape/3:output:04vae/nyan_encoder/ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:÷
#vae/nyan_encoder/ecc_conv_1/ReshapeReshape4vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd:output:02vae/nyan_encoder/ecc_conv_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  Ж
1vae/nyan_encoder/ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            И
3vae/nyan_encoder/ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            И
3vae/nyan_encoder/ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
+vae/nyan_encoder/ecc_conv_1/strided_slice_2StridedSlice$vae/nyan_encoder/ecc_conv_1/Cast:y:0:vae/nyan_encoder/ecc_conv_1/strided_slice_2/stack:output:0<vae/nyan_encoder/ecc_conv_1/strided_slice_2/stack_1:output:0<vae/nyan_encoder/ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_mask»
vae/nyan_encoder/ecc_conv_1/mulMul,vae/nyan_encoder/ecc_conv_1/Reshape:output:04vae/nyan_encoder/ecc_conv_1/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  ц
)vae/nyan_encoder/ecc_conv_1/einsum/EinsumEinsum#vae/nyan_encoder/ecc_conv_1/mul:z:0?vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdТ
#vae/nyan_encoder/ecc_conv_1/Shape_2Shape?vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:Л
#vae/nyan_encoder/ecc_conv_1/unstackUnpack,vae/nyan_encoder/ecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numЃ
2vae/nyan_encoder/ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0t
#vae/nyan_encoder/ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Л
%vae/nyan_encoder/ecc_conv_1/unstack_1Unpack,vae/nyan_encoder/ecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num|
+vae/nyan_encoder/ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ў
%vae/nyan_encoder/ecc_conv_1/Reshape_1Reshape?vae/nyan_encoder/ecc_conv/leaky_re_lu_1/LeakyRelu:activations:04vae/nyan_encoder/ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ∞
4vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0{
*vae/nyan_encoder/ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ќ
%vae/nyan_encoder/ecc_conv_1/transpose	Transpose<vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOp:value:03vae/nyan_encoder/ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  |
+vae/nyan_encoder/ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ї
%vae/nyan_encoder/ecc_conv_1/Reshape_2Reshape)vae/nyan_encoder/ecc_conv_1/transpose:y:04vae/nyan_encoder/ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Њ
"vae/nyan_encoder/ecc_conv_1/MatMulMatMul.vae/nyan_encoder/ecc_conv_1/Reshape_1:output:0.vae/nyan_encoder/ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
-vae/nyan_encoder/ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<o
-vae/nyan_encoder/ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : €
+vae/nyan_encoder/ecc_conv_1/Reshape_3/shapePack,vae/nyan_encoder/ecc_conv_1/unstack:output:06vae/nyan_encoder/ecc_conv_1/Reshape_3/shape/1:output:06vae/nyan_encoder/ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
: 
%vae/nyan_encoder/ecc_conv_1/Reshape_3Reshape,vae/nyan_encoder/ecc_conv_1/MatMul:product:04vae/nyan_encoder/ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ¬
vae/nyan_encoder/ecc_conv_1/addAddV22vae/nyan_encoder/ecc_conv_1/einsum/Einsum:output:0.vae/nyan_encoder/ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ™
2vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≈
#vae/nyan_encoder/ecc_conv_1/BiasAddBiasAdd#vae/nyan_encoder/ecc_conv_1/add:z:0:vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< ≈
!vae/nyan_encoder/ecc_conv_1/mul_1Mul,vae/nyan_encoder/ecc_conv_1/BiasAdd:output:07vae/nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< §
3vae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu	LeakyRelu%vae/nyan_encoder/ecc_conv_1/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=v
 vae/nyan_encoder/ecc_conv_2/CastCastinput_2*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<Т
!vae/nyan_encoder/ecc_conv_2/ShapeShapeAvae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:В
/vae/nyan_encoder/ecc_conv_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€Д
1vae/nyan_encoder/ecc_conv_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1vae/nyan_encoder/ecc_conv_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)vae/nyan_encoder/ecc_conv_2/strided_sliceStridedSlice*vae/nyan_encoder/ecc_conv_2/Shape:output:08vae/nyan_encoder/ecc_conv_2/strided_slice/stack:output:0:vae/nyan_encoder/ecc_conv_2/strided_slice/stack_1:output:0:vae/nyan_encoder/ecc_conv_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
#vae/nyan_encoder/ecc_conv_2/Shape_1ShapeAvae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:Д
1vae/nyan_encoder/ecc_conv_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3vae/nyan_encoder/ecc_conv_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3vae/nyan_encoder/ecc_conv_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+vae/nyan_encoder/ecc_conv_2/strided_slice_1StridedSlice,vae/nyan_encoder/ecc_conv_2/Shape_1:output:0:vae/nyan_encoder/ecc_conv_2/strided_slice_1/stack:output:0<vae/nyan_encoder/ecc_conv_2/strided_slice_1/stack_1:output:0<vae/nyan_encoder/ecc_conv_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask√
<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOpReadVariableOpEvae_nyan_encoder_ecc_conv_2_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0|
2vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:З
2vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          j
3vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ShapeShapeinput_3*
T0*
_output_shapes
:}
;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ћ
6vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2GatherV2<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:0;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:0Dvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѕ
8vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1GatherV2<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Shape:output:0;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0Fvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Џ
2vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ProdProd?vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: а
4vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1ProdAvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2_1:output:0>vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
4vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concatConcatV2;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/free:output:0;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/axes:output:0Bvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:е
3vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stackPack;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod:output:0=vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:∆
7vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose	Transposeinput_3=vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<ц
5vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReshapeReshape;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/transpose:y:0<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ч
4vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMulMatMul>vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Reshape:output:0Dvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АА
5vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А}
;vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
6vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1ConcatV2?vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/GatherV2:output:0>vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/Const_2:output:0Dvae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ф
-vae/nyan_encoder/ecc_conv_2/FGN_out/TensordotReshape>vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/MatMul:product:0?vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аї
:vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOpReadVariableOpCvae_nyan_encoder_ecc_conv_2_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0н
+vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAddBiasAdd6vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot:output:0Bvae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<Аv
+vae/nyan_encoder/ecc_conv_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€m
+vae/nyan_encoder/ecc_conv_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : й
)vae/nyan_encoder/ecc_conv_2/Reshape/shapePack4vae/nyan_encoder/ecc_conv_2/Reshape/shape/0:output:02vae/nyan_encoder/ecc_conv_2/strided_slice:output:02vae/nyan_encoder/ecc_conv_2/strided_slice:output:04vae/nyan_encoder/ecc_conv_2/Reshape/shape/3:output:04vae/nyan_encoder/ecc_conv_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:÷
#vae/nyan_encoder/ecc_conv_2/ReshapeReshape4vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd:output:02vae/nyan_encoder/ecc_conv_2/Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  Ж
1vae/nyan_encoder/ecc_conv_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            И
3vae/nyan_encoder/ecc_conv_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            И
3vae/nyan_encoder/ecc_conv_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
+vae/nyan_encoder/ecc_conv_2/strided_slice_2StridedSlice$vae/nyan_encoder/ecc_conv_2/Cast:y:0:vae/nyan_encoder/ecc_conv_2/strided_slice_2/stack:output:0<vae/nyan_encoder/ecc_conv_2/strided_slice_2/stack_1:output:0<vae/nyan_encoder/ecc_conv_2/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_mask»
vae/nyan_encoder/ecc_conv_2/mulMul,vae/nyan_encoder/ecc_conv_2/Reshape:output:04vae/nyan_encoder/ecc_conv_2/strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  ш
)vae/nyan_encoder/ecc_conv_2/einsum/EinsumEinsum#vae/nyan_encoder/ecc_conv_2/mul:z:0Avae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abdФ
#vae/nyan_encoder/ecc_conv_2/Shape_2ShapeAvae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:Л
#vae/nyan_encoder/ecc_conv_2/unstackUnpack,vae/nyan_encoder/ecc_conv_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numЃ
2vae/nyan_encoder/ecc_conv_2/Shape_3/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0t
#vae/nyan_encoder/ecc_conv_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        Л
%vae/nyan_encoder/ecc_conv_2/unstack_1Unpack,vae/nyan_encoder/ecc_conv_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num|
+vae/nyan_encoder/ecc_conv_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    џ
%vae/nyan_encoder/ecc_conv_2/Reshape_1ReshapeAvae/nyan_encoder/ecc_conv_1/leaky_re_lu_2/LeakyRelu:activations:04vae/nyan_encoder/ecc_conv_2/Reshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ∞
4vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_2_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype0{
*vae/nyan_encoder/ecc_conv_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ќ
%vae/nyan_encoder/ecc_conv_2/transpose	Transpose<vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOp:value:03vae/nyan_encoder/ecc_conv_2/transpose/perm:output:0*
T0*
_output_shapes

:  |
+vae/nyan_encoder/ecc_conv_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€Ї
%vae/nyan_encoder/ecc_conv_2/Reshape_2Reshape)vae/nyan_encoder/ecc_conv_2/transpose:y:04vae/nyan_encoder/ecc_conv_2/Reshape_2/shape:output:0*
T0*
_output_shapes

:  Њ
"vae/nyan_encoder/ecc_conv_2/MatMulMatMul.vae/nyan_encoder/ecc_conv_2/Reshape_1:output:0.vae/nyan_encoder/ecc_conv_2/Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ o
-vae/nyan_encoder/ecc_conv_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<o
-vae/nyan_encoder/ecc_conv_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : €
+vae/nyan_encoder/ecc_conv_2/Reshape_3/shapePack,vae/nyan_encoder/ecc_conv_2/unstack:output:06vae/nyan_encoder/ecc_conv_2/Reshape_3/shape/1:output:06vae/nyan_encoder/ecc_conv_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
: 
%vae/nyan_encoder/ecc_conv_2/Reshape_3Reshape,vae/nyan_encoder/ecc_conv_2/MatMul:product:04vae/nyan_encoder/ecc_conv_2/Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ¬
vae/nyan_encoder/ecc_conv_2/addAddV22vae/nyan_encoder/ecc_conv_2/einsum/Einsum:output:0.vae/nyan_encoder/ecc_conv_2/Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< ™
2vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOpReadVariableOp;vae_nyan_encoder_ecc_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≈
#vae/nyan_encoder/ecc_conv_2/BiasAddBiasAdd#vae/nyan_encoder/ecc_conv_2/add:z:0:vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< ≈
!vae/nyan_encoder/ecc_conv_2/mul_1Mul,vae/nyan_encoder/ecc_conv_2/BiasAdd:output:07vae/nyan_encoder/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€< §
3vae/nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu	LeakyRelu%vae/nyan_encoder/ecc_conv_2/mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=Б
6vae/nyan_encoder/global_sum_pool/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€б
$vae/nyan_encoder/global_sum_pool/SumSumAvae/nyan_encoder/ecc_conv_2/leaky_re_lu_3/LeakyRelu:activations:0?vae/nyan_encoder/global_sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ І
.vae/nyan_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp7vae_nyan_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 А*
dtype0√
vae/nyan_encoder/dense_1/MatMulMatMul-vae/nyan_encoder/global_sum_pool/Sum:output:06vae/nyan_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А•
/vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp8vae_nyan_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0¬
 vae/nyan_encoder/dense_1/BiasAddBiasAdd)vae/nyan_encoder/dense_1/MatMul:product:07vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АҐ
0vae/nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu	LeakyRelu)vae/nyan_encoder/dense_1/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=o
vae/nyan_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   «
 vae/nyan_encoder/flatten/ReshapeReshape>vae/nyan_encoder/dense_1/leaky_re_lu_4/LeakyRelu:activations:0'vae/nyan_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А®
.vae/nyan_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp7vae_nyan_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0њ
vae/nyan_encoder/dense_2/MatMulMatMul)vae/nyan_encoder/flatten/Reshape:output:06vae/nyan_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А•
/vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp8vae_nyan_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0¬
 vae/nyan_encoder/dense_2/BiasAddBiasAdd)vae/nyan_encoder/dense_2/MatMul:product:07vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АҐ
0vae/nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu	LeakyRelu)vae/nyan_encoder/dense_2/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=•
-vae/nyan_encoder/z_mean/MatMul/ReadVariableOpReadVariableOp6vae_nyan_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0—
vae/nyan_encoder/z_mean/MatMulMatMul>vae/nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:05vae/nyan_encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
.vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp7vae_nyan_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Њ
vae/nyan_encoder/z_mean/BiasAddBiasAdd(vae/nyan_encoder/z_mean/MatMul:product:06vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
0vae/nyan_encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp9vae_nyan_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0„
!vae/nyan_encoder/z_log_var/MatMulMatMul>vae/nyan_encoder/dense_2/leaky_re_lu_5/LeakyRelu:activations:08vae/nyan_encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@®
1vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp:vae_nyan_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0«
"vae/nyan_encoder/z_log_var/BiasAddBiasAdd+vae/nyan_encoder/z_log_var/MatMul:product:09vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@w
vae/nyan_encoder/sampling/ShapeShape(vae/nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:w
-vae/nyan_encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/vae/nyan_encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/vae/nyan_encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
'vae/nyan_encoder/sampling/strided_sliceStridedSlice(vae/nyan_encoder/sampling/Shape:output:06vae/nyan_encoder/sampling/strided_slice/stack:output:08vae/nyan_encoder/sampling/strided_slice/stack_1:output:08vae/nyan_encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
!vae/nyan_encoder/sampling/Shape_1Shape(vae/nyan_encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:y
/vae/nyan_encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1vae/nyan_encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1vae/nyan_encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)vae/nyan_encoder/sampling/strided_slice_1StridedSlice*vae/nyan_encoder/sampling/Shape_1:output:08vae/nyan_encoder/sampling/strided_slice_1/stack:output:0:vae/nyan_encoder/sampling/strided_slice_1/stack_1:output:0:vae/nyan_encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask…
-vae/nyan_encoder/sampling/random_normal/shapePack0vae/nyan_encoder/sampling/strided_slice:output:02vae/nyan_encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:q
,vae/nyan_encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    s
.vae/nyan_encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
<vae/nyan_encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal6vae/nyan_encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2а±£д
+vae/nyan_encoder/sampling/random_normal/mulMulEvae/nyan_encoder/sampling/random_normal/RandomStandardNormal:output:07vae/nyan_encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ 
'vae/nyan_encoder/sampling/random_normalAddV2/vae/nyan_encoder/sampling/random_normal/mul:z:05vae/nyan_encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
vae/nyan_encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?≠
vae/nyan_encoder/sampling/mulMul(vae/nyan_encoder/sampling/mul/x:output:0+vae/nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@y
vae/nyan_encoder/sampling/ExpExp!vae/nyan_encoder/sampling/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@®
vae/nyan_encoder/sampling/mul_1Mul!vae/nyan_encoder/sampling/Exp:y:0+vae/nyan_encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€@І
vae/nyan_encoder/sampling/addAddV2(vae/nyan_encoder/z_mean/BiasAdd:output:0#vae/nyan_encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@d
vae/nyan_encoder/sampling/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
!vae/nyan_encoder/sampling/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Т
(vae/nyan_encoder/sampling/ReadVariableOpReadVariableOp1vae_nyan_encoder_sampling_readvariableop_resource*
_output_shapes
: *
dtype0f
!vae/nyan_encoder/sampling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<•
vae/nyan_encoder/sampling/mul_2Mul*vae/nyan_encoder/sampling/mul_2/x:output:00vae/nyan_encoder/sampling/ReadVariableOp:value:0*
T0*
_output_shapes
: d
vae/nyan_encoder/sampling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>Ф
vae/nyan_encoder/sampling/subSub#vae/nyan_encoder/sampling/mul_2:z:0(vae/nyan_encoder/sampling/sub/y:output:0*
T0*
_output_shapes
: Ь
!vae/nyan_encoder/sampling/MaximumMaximum*vae/nyan_encoder/sampling/Const_1:output:0!vae/nyan_encoder/sampling/sub:z:0*
T0*
_output_shapes
: Ю
!vae/nyan_encoder/sampling/MinimumMinimum(vae/nyan_encoder/sampling/Const:output:0%vae/nyan_encoder/sampling/Maximum:z:0*
T0*
_output_shapes
: f
!vae/nyan_encoder/sampling/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≥
vae/nyan_encoder/sampling/add_1AddV2*vae/nyan_encoder/sampling/add_1/x:output:0+vae/nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
 vae/nyan_encoder/sampling/SquareSquare(vae/nyan_encoder/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@£
vae/nyan_encoder/sampling/sub_1Sub#vae/nyan_encoder/sampling/add_1:z:0$vae/nyan_encoder/sampling/Square:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
vae/nyan_encoder/sampling/Exp_1Exp+vae/nyan_encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
vae/nyan_encoder/sampling/sub_2Sub#vae/nyan_encoder/sampling/sub_1:z:0#vae/nyan_encoder/sampling/Exp_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€@f
!vae/nyan_encoder/sampling/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   њ©
vae/nyan_encoder/sampling/mul_3Mul*vae/nyan_encoder/sampling/mul_3/x:output:0#vae/nyan_encoder/sampling/sub_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@q
/vae/nyan_encoder/sampling/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :±
vae/nyan_encoder/sampling/SumSum#vae/nyan_encoder/sampling/mul_3:z:08vae/nyan_encoder/sampling/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:€€€€€€€€€k
!vae/nyan_encoder/sampling/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Ы
vae/nyan_encoder/sampling/MeanMean&vae/nyan_encoder/sampling/Sum:output:0*vae/nyan_encoder/sampling/Const_2:output:0*
T0*
_output_shapes
: Ч
vae/nyan_encoder/sampling/mul_4Mul%vae/nyan_encoder/sampling/Minimum:z:0'vae/nyan_encoder/sampling/Mean:output:0*
T0*
_output_shapes
: h
#vae/nyan_encoder/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * `jF†
!vae/nyan_encoder/sampling/truedivRealDiv#vae/nyan_encoder/sampling/mul_4:z:0,vae/nyan_encoder/sampling/truediv/y:output:0*
T0*
_output_shapes
: `
vae/nyan_encoder/sampling/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%vae/nyan_encoder/sampling/range/startConst*
_output_shapes
: *
dtype0*
value	B : g
%vae/nyan_encoder/sampling/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ћ
vae/nyan_encoder/sampling/rangeRange.vae/nyan_encoder/sampling/range/start:output:0'vae/nyan_encoder/sampling/Rank:output:0.vae/nyan_encoder/sampling/range/delta:output:0*
_output_shapes
: Ъ
vae/nyan_encoder/sampling/Sum_1Sum'vae/nyan_encoder/sampling/Mean:output:0(vae/nyan_encoder/sampling/range:output:0*
T0*
_output_shapes
: …
-vae/nyan_encoder/sampling/AssignAddVariableOpAssignAddVariableOp6vae_nyan_encoder_sampling_assignaddvariableop_resource(vae/nyan_encoder/sampling/Sum_1:output:0*
_output_shapes
 *
dtype0`
vae/nyan_encoder/sampling/SizeConst*
_output_shapes
: *
dtype0*
value	B :
vae/nyan_encoder/sampling/CastCast'vae/nyan_encoder/sampling/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ч
/vae/nyan_encoder/sampling/AssignAddVariableOp_1AssignAddVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_1_resource"vae/nyan_encoder/sampling/Cast:y:0.^vae/nyan_encoder/sampling/AssignAddVariableOp*
_output_shapes
 *
dtype0Д
3vae/nyan_encoder/sampling/div_no_nan/ReadVariableOpReadVariableOp6vae_nyan_encoder_sampling_assignaddvariableop_resource.^vae/nyan_encoder/sampling/AssignAddVariableOp0^vae/nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ў
5vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_1ReadVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_1_resource0^vae/nyan_encoder/sampling/AssignAddVariableOp_1*
_output_shapes
: *
dtype0Ќ
$vae/nyan_encoder/sampling/div_no_nanDivNoNan;vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp:value:0=vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: y
"vae/nyan_encoder/sampling/IdentityIdentity(vae/nyan_encoder/sampling/div_no_nan:z:0*
T0*
_output_shapes
: b
 vae/nyan_encoder/sampling/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : i
'vae/nyan_encoder/sampling/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : i
'vae/nyan_encoder/sampling/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :”
!vae/nyan_encoder/sampling/range_1Range0vae/nyan_encoder/sampling/range_1/start:output:0)vae/nyan_encoder/sampling/Rank_1:output:00vae/nyan_encoder/sampling/range_1/delta:output:0*
_output_shapes
: Ъ
vae/nyan_encoder/sampling/Sum_2Sum%vae/nyan_encoder/sampling/truediv:z:0*vae/nyan_encoder/sampling/range_1:output:0*
T0*
_output_shapes
: Ќ
/vae/nyan_encoder/sampling/AssignAddVariableOp_2AssignAddVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_2_resource(vae/nyan_encoder/sampling/Sum_2:output:0*
_output_shapes
 *
dtype0b
 vae/nyan_encoder/sampling/Size_1Const*
_output_shapes
: *
dtype0*
value	B :Г
 vae/nyan_encoder/sampling/Cast_1Cast)vae/nyan_encoder/sampling/Size_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ы
/vae/nyan_encoder/sampling/AssignAddVariableOp_3AssignAddVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_3_resource$vae/nyan_encoder/sampling/Cast_1:y:00^vae/nyan_encoder/sampling/AssignAddVariableOp_2*
_output_shapes
 *
dtype0К
5vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOpReadVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_2_resource0^vae/nyan_encoder/sampling/AssignAddVariableOp_20^vae/nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0Џ
7vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1ReadVariableOp8vae_nyan_encoder_sampling_assignaddvariableop_3_resource0^vae/nyan_encoder/sampling/AssignAddVariableOp_3*
_output_shapes
: *
dtype0”
&vae/nyan_encoder/sampling/div_no_nan_1DivNoNan=vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp:value:0?vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: }
$vae/nyan_encoder/sampling/Identity_1Identity*vae/nyan_encoder/sampling/div_no_nan_1:z:0*
T0*
_output_shapes
: І
.vae/nyan_decoder/dense_3/MatMul/ReadVariableOpReadVariableOp7vae_nyan_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Ј
vae/nyan_decoder/dense_3/MatMulMatMul!vae/nyan_encoder/sampling/add:z:06vae/nyan_decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А•
/vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp8vae_nyan_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0¬
 vae/nyan_decoder/dense_3/BiasAddBiasAdd)vae/nyan_decoder/dense_3/MatMul:product:07vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АҐ
0vae/nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu	LeakyRelu)vae/nyan_decoder/dense_3/BiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=®
.vae/nyan_decoder/dense_4/MatMul/ReadVariableOpReadVariableOp7vae_nyan_decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0‘
vae/nyan_decoder/dense_4/MatMulMatMul>vae/nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:06vae/nyan_decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€І•
/vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp8vae_nyan_decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0¬
 vae/nyan_decoder/dense_4/BiasAddBiasAdd)vae/nyan_decoder/dense_4/MatMul:product:07vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІЙ
 vae/nyan_decoder/dense_4/SigmoidSigmoid)vae/nyan_decoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€І®
.vae/nyan_decoder/dense_5/MatMul/ReadVariableOpReadVariableOp7vae_nyan_decoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0‘
vae/nyan_decoder/dense_5/MatMulMatMul>vae/nyan_decoder/dense_3/leaky_re_lu_6/LeakyRelu:activations:06vae/nyan_decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќ•
/vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp8vae_nyan_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0¬
 vae/nyan_decoder/dense_5/BiasAddBiasAdd)vae/nyan_decoder/dense_5/MatMul:product:07vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќg
vae/nyan_decoder/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€„
vae/nyan_decoder/concatConcatV2$vae/nyan_decoder/dense_4/Sigmoid:y:0)vae/nyan_decoder/dense_5/BiasAdd:output:0%vae/nyan_decoder/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€фp
IdentityIdentity vae/nyan_decoder/concat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фа
NoOpNoOp0^vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOp/^vae/nyan_decoder/dense_3/MatMul/ReadVariableOp0^vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOp/^vae/nyan_decoder/dense_4/MatMul/ReadVariableOp0^vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOp/^vae/nyan_decoder/dense_5/MatMul/ReadVariableOp.^vae/nyan_encoder/dense/BiasAdd/ReadVariableOp0^vae/nyan_encoder/dense/Tensordot/ReadVariableOp0^vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOp/^vae/nyan_encoder/dense_1/MatMul/ReadVariableOp0^vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOp/^vae/nyan_encoder/dense_2/MatMul/ReadVariableOp1^vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp9^vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp;^vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp3^vae/nyan_encoder/ecc_conv/transpose/ReadVariableOp3^vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp;^vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp=^vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp5^vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOp3^vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp;^vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp=^vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp5^vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOp.^vae/nyan_encoder/sampling/AssignAddVariableOp0^vae/nyan_encoder/sampling/AssignAddVariableOp_10^vae/nyan_encoder/sampling/AssignAddVariableOp_20^vae/nyan_encoder/sampling/AssignAddVariableOp_3)^vae/nyan_encoder/sampling/ReadVariableOp4^vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp6^vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_16^vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp8^vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_12^vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOp1^vae/nyan_encoder/z_log_var/MatMul/ReadVariableOp/^vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOp.^vae/nyan_encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOp/vae/nyan_decoder/dense_3/BiasAdd/ReadVariableOp2`
.vae/nyan_decoder/dense_3/MatMul/ReadVariableOp.vae/nyan_decoder/dense_3/MatMul/ReadVariableOp2b
/vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOp/vae/nyan_decoder/dense_4/BiasAdd/ReadVariableOp2`
.vae/nyan_decoder/dense_4/MatMul/ReadVariableOp.vae/nyan_decoder/dense_4/MatMul/ReadVariableOp2b
/vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOp/vae/nyan_decoder/dense_5/BiasAdd/ReadVariableOp2`
.vae/nyan_decoder/dense_5/MatMul/ReadVariableOp.vae/nyan_decoder/dense_5/MatMul/ReadVariableOp2^
-vae/nyan_encoder/dense/BiasAdd/ReadVariableOp-vae/nyan_encoder/dense/BiasAdd/ReadVariableOp2b
/vae/nyan_encoder/dense/Tensordot/ReadVariableOp/vae/nyan_encoder/dense/Tensordot/ReadVariableOp2b
/vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOp/vae/nyan_encoder/dense_1/BiasAdd/ReadVariableOp2`
.vae/nyan_encoder/dense_1/MatMul/ReadVariableOp.vae/nyan_encoder/dense_1/MatMul/ReadVariableOp2b
/vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOp/vae/nyan_encoder/dense_2/BiasAdd/ReadVariableOp2`
.vae/nyan_encoder/dense_2/MatMul/ReadVariableOp.vae/nyan_encoder/dense_2/MatMul/ReadVariableOp2d
0vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp0vae/nyan_encoder/ecc_conv/BiasAdd/ReadVariableOp2t
8vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp8vae/nyan_encoder/ecc_conv/FGN_out/BiasAdd/ReadVariableOp2x
:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp:vae/nyan_encoder/ecc_conv/FGN_out/Tensordot/ReadVariableOp2h
2vae/nyan_encoder/ecc_conv/transpose/ReadVariableOp2vae/nyan_encoder/ecc_conv/transpose/ReadVariableOp2h
2vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp2vae/nyan_encoder/ecc_conv_1/BiasAdd/ReadVariableOp2x
:vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:vae/nyan_encoder/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2|
<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp<vae/nyan_encoder/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2l
4vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOp4vae/nyan_encoder/ecc_conv_1/transpose/ReadVariableOp2h
2vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp2vae/nyan_encoder/ecc_conv_2/BiasAdd/ReadVariableOp2x
:vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp:vae/nyan_encoder/ecc_conv_2/FGN_out/BiasAdd/ReadVariableOp2|
<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp<vae/nyan_encoder/ecc_conv_2/FGN_out/Tensordot/ReadVariableOp2l
4vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOp4vae/nyan_encoder/ecc_conv_2/transpose/ReadVariableOp2^
-vae/nyan_encoder/sampling/AssignAddVariableOp-vae/nyan_encoder/sampling/AssignAddVariableOp2b
/vae/nyan_encoder/sampling/AssignAddVariableOp_1/vae/nyan_encoder/sampling/AssignAddVariableOp_12b
/vae/nyan_encoder/sampling/AssignAddVariableOp_2/vae/nyan_encoder/sampling/AssignAddVariableOp_22b
/vae/nyan_encoder/sampling/AssignAddVariableOp_3/vae/nyan_encoder/sampling/AssignAddVariableOp_32T
(vae/nyan_encoder/sampling/ReadVariableOp(vae/nyan_encoder/sampling/ReadVariableOp2j
3vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp3vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp2n
5vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_15vae/nyan_encoder/sampling/div_no_nan/ReadVariableOp_12n
5vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp5vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp2r
7vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_17vae/nyan_encoder/sampling/div_no_nan_1/ReadVariableOp_12f
1vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOp1vae/nyan_encoder/z_log_var/BiasAdd/ReadVariableOp2d
0vae/nyan_encoder/z_log_var/MatMul/ReadVariableOp0vae/nyan_encoder/z_log_var/MatMul/ReadVariableOp2`
.vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOp.vae/nyan_encoder/z_mean/BiasAdd/ReadVariableOp2^
-vae/nyan_encoder/z_mean/MatMul/ReadVariableOp-vae/nyan_encoder/z_mean/MatMul/ReadVariableOp:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
Д	
Р
.__inference_nyan_decoder_layer_call_fn_7408660
input_1
unknown:	@А
	unknown_0:	А
	unknown_1:
АІ
	unknown_2:	І
	unknown_3:
АЌ
	unknown_4:	Ќ
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€@
!
_user_specified_name	input_1
Е
Э
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408728
input_1"
dense_3_7408710:	@А
dense_3_7408712:	А#
dense_4_7408715:
АІ
dense_4_7408717:	І#
dense_5_7408720:
АЌ
dense_5_7408722:	Ќ
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallф
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_7408710dense_3_7408712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_7408603Х
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_7408715dense_4_7408717*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€І*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_7408620Х
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_5_7408720dense_5_7408722*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ќ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_7408636V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Є
concatConcatV2(dense_4/StatefulPartitionedCall:output:0(dense_5/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ф_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€@
!
_user_specified_name	input_1
ЩE
√

I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408585
x
a	
e
dense_7408517:
dense_7408519:#
ecc_conv_7408522:	А
ecc_conv_7408524:	А"
ecc_conv_7408526: 
ecc_conv_7408528: %
ecc_conv_1_7408531:	А!
ecc_conv_1_7408533:	А$
ecc_conv_1_7408535:   
ecc_conv_1_7408537: %
ecc_conv_2_7408540:	А!
ecc_conv_2_7408542:	А$
ecc_conv_2_7408544:   
ecc_conv_2_7408546: "
dense_1_7408550:	 А
dense_1_7408552:	А#
dense_2_7408556:
АА
dense_2_7408558:	А!
z_mean_7408561:	А@
z_mean_7408563:@$
z_log_var_7408566:	А@
z_log_var_7408568:@
sampling_7408571: 
sampling_7408573: 
sampling_7408575: 
sampling_7408577: 
sampling_7408579: 
identity

identity_1ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ ecc_conv/StatefulPartitionedCallҐ"ecc_conv_1/StatefulPartitionedCallҐ"ecc_conv_2/StatefulPartitionedCallҐ sampling/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall≈
graph_masking/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7408162r
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ю
graph_masking/strided_sliceStridedSlicex*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskО
dense/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0dense_7408517dense_7408519*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7407503с
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_7408522ecc_conv_7408524ecc_conv_7408526ecc_conv_7408528*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590А
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_1_7408531ecc_conv_1_7408533ecc_conv_1_7408535ecc_conv_1_7408537*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681В
"ecc_conv_2/StatefulPartitionedCallStatefulPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0ae$graph_masking/strided_slice:output:0ecc_conv_2_7408540ecc_conv_2_7408542ecc_conv_2_7408544ecc_conv_2_7408546*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772п
global_sum_pool/PartitionedCallPartitionedCall+ecc_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788Х
dense_1/StatefulPartitionedCallStatefulPartitionedCall(global_sum_pool/PartitionedCall:output:0dense_1_7408550dense_1_7408552*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801Ё
flatten/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7407813Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_7408556dense_2_7408558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826Р
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_7408561z_mean_7408563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842Ь
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_7408566z_log_var_7408568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858€
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0sampling_7408571sampling_7408573sampling_7408575sampling_7408577sampling_7408579*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_7407942x
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@i

Identity_1Identity)sampling/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: €
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall#^ecc_conv_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall2H
"ecc_conv_2/StatefulPartitionedCall"ecc_conv_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:N J
+
_output_shapes
:€€€€€€€€€<

_user_specified_namex:NJ
+
_output_shapes
:€€€€€€€€€<<

_user_specified_namea:RN
/
_output_shapes
:€€€€€€€€€<<

_user_specified_namee
’

ч
D__inference_dense_3_layer_call_and_return_conditional_losses_7408603

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_6/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
њL
№
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772

inputs
inputs_1	
inputs_2
mask<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  Е
einsum/EinsumEinsummul:z:0inputs*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd=
Shape_2Shapeinputs*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    h
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Z
mul_1MulBiasAdd:output:0mask*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_3/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€< 
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:QM
+
_output_shapes
:€€€€€€€€€<

_user_specified_namemask
ў

ш
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_5/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
…
Ш
)__inference_dense_3_layer_call_fn_7411794

inputs
unknown:	@А
	unknown_0:	А
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_7408603p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ѕ
щ
B__inference_dense_layer_call_and_return_conditional_losses_7407503

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€<К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€<q
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*+
_output_shapes
:€€€€€€€€€<*
alpha%ЌћL=v
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
ћ
Щ
+__inference_z_log_var_layer_call_fn_7411679

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
“	
ш
D__inference_dense_5_layer_call_and_return_conditional_losses_7411844

inputs2
matmul_readvariableop_resource:
АЌ.
biasadd_readvariableop_resource:	Ќ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АЌ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ќ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Ќw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶

ш
D__inference_dense_4_layer_call_and_return_conditional_losses_7408620

inputs2
matmul_readvariableop_resource:
АІ.
biasadd_readvariableop_resource:	І
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АІ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Іs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:І*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ІW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€І[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Іw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ў

ш
D__inference_dense_2_layer_call_and_return_conditional_losses_7411651

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
leaky_re_lu_5/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:€€€€€€€€€А*
alpha%ЌћL=u
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ф
µ
@__inference_vae_layer_call_and_return_conditional_losses_7409330
input_1
input_2	
input_3&
nyan_encoder_7409259:"
nyan_encoder_7409261:'
nyan_encoder_7409263:	А#
nyan_encoder_7409265:	А&
nyan_encoder_7409267: "
nyan_encoder_7409269: '
nyan_encoder_7409271:	А#
nyan_encoder_7409273:	А&
nyan_encoder_7409275:  "
nyan_encoder_7409277: '
nyan_encoder_7409279:	А#
nyan_encoder_7409281:	А&
nyan_encoder_7409283:  "
nyan_encoder_7409285: '
nyan_encoder_7409287:	 А#
nyan_encoder_7409289:	А(
nyan_encoder_7409291:
АА#
nyan_encoder_7409293:	А'
nyan_encoder_7409295:	А@"
nyan_encoder_7409297:@'
nyan_encoder_7409299:	А@"
nyan_encoder_7409301:@
nyan_encoder_7409303: 
nyan_encoder_7409305: 
nyan_encoder_7409307: 
nyan_encoder_7409309: 
nyan_encoder_7409311: '
nyan_decoder_7409315:	@А#
nyan_decoder_7409317:	А(
nyan_decoder_7409319:
АІ#
nyan_decoder_7409321:	І(
nyan_decoder_7409323:
АЌ#
nyan_decoder_7409325:	Ќ
identity

identity_1ИҐ$nyan_decoder/StatefulPartitionedCallҐ$nyan_encoder/StatefulPartitionedCallт
$nyan_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3nyan_encoder_7409259nyan_encoder_7409261nyan_encoder_7409263nyan_encoder_7409265nyan_encoder_7409267nyan_encoder_7409269nyan_encoder_7409271nyan_encoder_7409273nyan_encoder_7409275nyan_encoder_7409277nyan_encoder_7409279nyan_encoder_7409281nyan_encoder_7409283nyan_encoder_7409285nyan_encoder_7409287nyan_encoder_7409289nyan_encoder_7409291nyan_encoder_7409293nyan_encoder_7409295nyan_encoder_7409297nyan_encoder_7409299nyan_encoder_7409301nyan_encoder_7409303nyan_encoder_7409305nyan_encoder_7409307nyan_encoder_7409309nyan_encoder_7409311*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408309О
$nyan_decoder/StatefulPartitionedCallStatefulPartitionedCall-nyan_encoder/StatefulPartitionedCall:output:0nyan_decoder_7409315nyan_decoder_7409317nyan_decoder_7409319nyan_decoder_7409321nyan_decoder_7409323nyan_decoder_7409325*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408645}
IdentityIdentity-nyan_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€фm

Identity_1Identity-nyan_encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ф
NoOpNoOp%^nyan_decoder/StatefulPartitionedCall%^nyan_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$nyan_decoder/StatefulPartitionedCall$nyan_decoder/StatefulPartitionedCall2L
$nyan_encoder/StatefulPartitionedCall$nyan_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
Х
€
%__inference_vae_layer_call_fn_7409178
input_1
input_2	
input_3
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26:	@А

unknown_27:	А

unknown_28:
АІ

unknown_29:	І

unknown_30:
АЌ

unknown_31:	Ќ
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:€€€€€€€€€ф: *?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vae_layer_call_and_return_conditional_losses_7409034p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:€€€€€€€€€<
!
_user_specified_name	input_1:TP
+
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_2:XT
/
_output_shapes
:€€€€€€€€€<<
!
_user_specified_name	input_3
Љ
f
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411250

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
Є
`
D__inference_flatten_layer_call_and_return_conditional_losses_7407813

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
f
J__inference_graph_masking_layer_call_and_return_conditional_losses_7407466

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
њL
№
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681

inputs
inputs_1	
inputs_2
mask<
)fgn_out_tensordot_readvariableop_resource:	А6
'fgn_out_biasadd_readvariableop_resource:	А1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐFGN_out/BiasAdd/ReadVariableOpҐ FGN_out/Tensordot/ReadVariableOpҐtranspose/ReadVariableOp[
CastCastinputs_1*

DstT0*

SrcT0	*+
_output_shapes
:€€€€€€€€€<<;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype0`
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          O
FGN_out/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:a
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: c
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Љ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
FGN_out/Tensordot/transpose	Transposeinputs_2!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€<<Ґ
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€£
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аa
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:†
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€<<АГ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€<<АZ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ѕ
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:В
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         В
strided_slice_2StridedSliceCast:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€<<*
ellipsis_mask*
new_axis_maskt
mulMulReshape:output:0strided_slice_2:output:0*
T0*3
_output_shapes!
:€€€€€€€€€<<  Е
einsum/EinsumEinsummul:z:0inputs*
N*
T0*+
_output_shapes
:€€€€€€€€€< *
equationabcde,ace->abd=
Shape_2Shapeinputs*
T0*
_output_shapes
:S
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    h
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ x
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    €€€€f
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  j
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ S
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<S
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : П
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€< n
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:€€€€€€€€€< r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0q
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€< Z
mul_1MulBiasAdd:output:0mask*
T0*+
_output_shapes
:€€€€€€€€€< l
leaky_re_lu_2/LeakyRelu	LeakyRelu	mul_1:z:0*+
_output_shapes
:€€€€€€€€€< *
alpha%ЌћL=x
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€< Њ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:€€€€€€€€€< :€€€€€€€€€<<:€€€€€€€€€<<:€€€€€€€€€<: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€< 
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:QM
+
_output_shapes
:€€€€€€€€€<

_user_specified_namemask
…
Ш
)__inference_dense_1_layer_call_fn_7411609

inputs
unknown:	 А
	unknown_0:	А
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Э
џ
.__inference_nyan_encoder_layer_call_fn_7410457
inputs_0
inputs_1	
inputs_2
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 
identityИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
Ю
В
%__inference_vae_layer_call_fn_7409483
inputs_0
inputs_1	
inputs_2
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26:	@А

unknown_27:	А

unknown_28:
АІ

unknown_29:	І

unknown_30:
АЌ

unknown_31:	Ќ
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:€€€€€€€€€ф: *?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vae_layer_call_and_return_conditional_losses_7408810p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
Љ
f
J__inference_graph_masking_layer_call_and_return_conditional_losses_7408162

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
“
h
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788

inputs
identity`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€d
SumSuminputsSum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€< :S O
+
_output_shapes
:€€€€€€€€€< 
 
_user_specified_nameinputs
—
Ф
'__inference_dense_layer_call_fn_7411267

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7407503s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€<: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
пE
÷

I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408309

inputs
inputs_1	
inputs_2
dense_7408241:
dense_7408243:#
ecc_conv_7408246:	А
ecc_conv_7408248:	А"
ecc_conv_7408250: 
ecc_conv_7408252: %
ecc_conv_1_7408255:	А!
ecc_conv_1_7408257:	А$
ecc_conv_1_7408259:   
ecc_conv_1_7408261: %
ecc_conv_2_7408264:	А!
ecc_conv_2_7408266:	А$
ecc_conv_2_7408268:   
ecc_conv_2_7408270: "
dense_1_7408274:	 А
dense_1_7408276:	А#
dense_2_7408280:
АА
dense_2_7408282:	А!
z_mean_7408285:	А@
z_mean_7408287:@$
z_log_var_7408290:	А@
z_log_var_7408292:@
sampling_7408295: 
sampling_7408297: 
sampling_7408299: 
sampling_7408301: 
sampling_7408303: 
identity

identity_1ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ ecc_conv/StatefulPartitionedCallҐ"ecc_conv_1/StatefulPartitionedCallҐ"ecc_conv_2/StatefulPartitionedCallҐ sampling/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall 
graph_masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_graph_masking_layer_call_and_return_conditional_losses_7408162r
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    €€€€t
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      £
graph_masking/strided_sliceStridedSliceinputs*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*
ellipsis_mask*
end_maskО
dense/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0dense_7408241dense_7408243*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7407503€
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_7408246ecc_conv_7408248ecc_conv_7408250ecc_conv_7408252*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7407590О
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_1_7408255ecc_conv_1_7408257ecc_conv_1_7408259ecc_conv_1_7408261*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7407681Р
"ecc_conv_2/StatefulPartitionedCallStatefulPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_2_7408264ecc_conv_2_7408266ecc_conv_2_7408268ecc_conv_2_7408270*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€< *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7407772п
global_sum_pool/PartitionedCallPartitionedCall+ecc_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7407788Х
dense_1/StatefulPartitionedCallStatefulPartitionedCall(global_sum_pool/PartitionedCall:output:0dense_1_7408274dense_1_7408276*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7407801Ё
flatten/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7407813Н
dense_2/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_2_7408280dense_2_7408282*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7407826Р
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_7408285z_mean_7408287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_7407842Ь
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_7408290z_log_var_7408292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_7407858€
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0sampling_7408295sampling_7408297sampling_7408299sampling_7408301sampling_7408303*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€@: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_7407942x
IdentityIdentity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@i

Identity_1Identity)sampling/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: €
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall#^ecc_conv_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall2H
"ecc_conv_2/StatefulPartitionedCall"ecc_conv_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€<<
 
_user_specified_nameinputs
Ю
В
%__inference_vae_layer_call_fn_7409557
inputs_0
inputs_1	
inputs_2
unknown:
	unknown_0:
	unknown_1:	А
	unknown_2:	А
	unknown_3: 
	unknown_4: 
	unknown_5:	А
	unknown_6:	А
	unknown_7:  
	unknown_8: 
	unknown_9:	А

unknown_10:	А

unknown_11:  

unknown_12: 

unknown_13:	 А

unknown_14:	А

unknown_15:
АА

unknown_16:	А

unknown_17:	А@

unknown_18:@

unknown_19:	А@

unknown_20:@

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26:	@А

unknown_27:	А

unknown_28:
АІ

unknown_29:	І

unknown_30:
АЌ

unknown_31:	Ќ
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:€€€€€€€€€ф: *?
_read_only_resource_inputs!
	
 !"#*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vae_layer_call_and_return_conditional_losses_7409034p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ф`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:€€€€€€€€€<:€€€€€€€€€<<:€€€€€€€€€<<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:€€€€€€€€€<
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:€€€€€€€€€<<
"
_user_specified_name
inputs/2
Љ
f
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411258

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    €€€€f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€<*

begin_mask*
ellipsis_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€<:S O
+
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ґ
serving_defaultҐ
?
input_14
serving_default_input_1:0€€€€€€€€€<
?
input_24
serving_default_input_2:0	€€€€€€€€€<<
C
input_38
serving_default_input_3:0€€€€€€€€€<<=
output_11
StatefulPartitionedCall:0€€€€€€€€€фtensorflow/serving/predict:еЊ
З
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder


epochs
	optimizer

signatures"
_tf_keras_model
Ж
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21

22
#23
$24
%25
&26
'27
(28
)29"
trackable_list_wrapper
ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24
&25
'26
(27"
trackable_list_wrapper
 "
trackable_list_wrapper
 
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
…
/trace_0
0trace_1
1trace_2
2trace_32ё
%__inference_vae_layer_call_fn_7408880
%__inference_vae_layer_call_fn_7409483
%__inference_vae_layer_call_fn_7409557
%__inference_vae_layer_call_fn_7409178њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 z/trace_0z0trace_1z1trace_2z2trace_3
µ
3trace_0
4trace_1
5trace_2
6trace_32 
@__inference_vae_layer_call_and_return_conditional_losses_7409945
@__inference_vae_layer_call_and_return_conditional_losses_7410333
@__inference_vae_layer_call_and_return_conditional_losses_7409254
@__inference_vae_layer_call_and_return_conditional_losses_7409330њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 z3trace_0z4trace_1z5trace_2z6trace_3
яB№
"__inference__wrapped_model_7407447input_1input_2input_3"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Н
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=input_names
>output_names


epochs
?masking
@precondition
A
graphconv1
B
graphconv2
C
graphconv3
	Dpool1

Edense1
Fflatten

Gdense2

Hz_mean
I	z_log_var
Jlatent_z
Knyan_layers"
_tf_keras_model
п
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

)epochs

Rdense3
Sfingerprint
T
regression
Unyan_layers"
_tf_keras_model
: 2global_step
р
Viter

Wbeta_1

Xbeta_2
	Ydecaym€mАmБmВmГmДmЕmЖmЗmИmЙmКmЛmМmНmОmПmРmС mТ!mУ"mФ#mХ$mЦ%mЧ&mШ'mЩ(mЪvЫvЬvЭvЮvЯv†v°vҐv£v§v•v¶vІv®v©v™vЂvђv≠ vЃ!vѓ"v∞#v±$v≤%v≥&vі'vµ(vґ"
	optimizer
,
Zserving_default"
signature_map
/:-2vae/nyan_encoder/dense/kernel
):'2vae/nyan_encoder/dense/bias
7:5 2%vae/nyan_encoder/ecc_conv/root_kernel
,:* 2vae/nyan_encoder/ecc_conv/bias
;:9	А2(vae/nyan_encoder/ecc_conv/FGN_out/kernel
5:3А2&vae/nyan_encoder/ecc_conv/FGN_out/bias
9:7  2'vae/nyan_encoder/ecc_conv_1/root_kernel
.:, 2 vae/nyan_encoder/ecc_conv_1/bias
=:;	А2*vae/nyan_encoder/ecc_conv_1/FGN_out/kernel
7:5А2(vae/nyan_encoder/ecc_conv_1/FGN_out/bias
9:7  2'vae/nyan_encoder/ecc_conv_2/root_kernel
.:, 2 vae/nyan_encoder/ecc_conv_2/bias
=:;	А2*vae/nyan_encoder/ecc_conv_2/FGN_out/kernel
7:5А2(vae/nyan_encoder/ecc_conv_2/FGN_out/bias
2:0	 А2vae/nyan_encoder/dense_1/kernel
,:*А2vae/nyan_encoder/dense_1/bias
3:1
АА2vae/nyan_encoder/dense_2/kernel
,:*А2vae/nyan_encoder/dense_2/bias
1:/	А@2vae/nyan_encoder/z_mean/kernel
*:(@2vae/nyan_encoder/z_mean/bias
4:2	А@2!vae/nyan_encoder/z_log_var/kernel
-:+@2vae/nyan_encoder/z_log_var/bias
2:0	@А2vae/nyan_decoder/dense_3/kernel
,:*А2vae/nyan_decoder/dense_3/bias
3:1
АІ2vae/nyan_decoder/dense_4/kernel
,:*І2vae/nyan_decoder/dense_4/bias
3:1
АЌ2vae/nyan_decoder/dense_5/kernel
,:*Ќ2vae/nyan_decoder/dense_5/bias
: 2global_step
.

0
)1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
C
[0
\1
]2
^3
_4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЙBЖ
%__inference_vae_layer_call_fn_7408880input_1input_2input_3"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
МBЙ
%__inference_vae_layer_call_fn_7409483inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
МBЙ
%__inference_vae_layer_call_fn_7409557inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЙBЖ
%__inference_vae_layer_call_fn_7409178input_1input_2input_3"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ІB§
@__inference_vae_layer_call_and_return_conditional_losses_7409945inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ІB§
@__inference_vae_layer_call_and_return_conditional_losses_7410333inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
§B°
@__inference_vae_layer_call_and_return_conditional_losses_7409254input_1input_2input_3"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
§B°
@__inference_vae_layer_call_and_return_conditional_losses_7409330input_1input_2input_3"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ќ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21

22"
trackable_list_wrapper
∆
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
н
etrace_0
ftrace_1
gtrace_2
htrace_32В
.__inference_nyan_encoder_layer_call_fn_7408015
.__inference_nyan_encoder_layer_call_fn_7410395
.__inference_nyan_encoder_layer_call_fn_7410457
.__inference_nyan_encoder_layer_call_fn_7408429њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 zetrace_0zftrace_1zgtrace_2zhtrace_3
ў
itrace_0
jtrace_1
ktrace_2
ltrace_32о
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7410823
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7411189
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408507
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408585њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 zitrace_0zjtrace_1zktrace_2zltrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
•
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y
activation

kernel
bias"
_tf_keras_layer
€
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkwargs_keys
Б
activation
Вkernel_network_layers
root_kernel
bias"
_tf_keras_layer
Е
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkwargs_keys
К
activation
Лkernel_network_layers
root_kernel
bias"
_tf_keras_layer
Е
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkwargs_keys
У
activation
Фkernel_network_layers
root_kernel
bias"
_tf_keras_layer
Ђ
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
“
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°
activation

kernel
bias"
_tf_keras_layer
Ђ
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
“
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses
Ѓ
activation

kernel
bias"
_tf_keras_layer
Ѕ
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
Ѕ
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
Ђ
ї	variables
Љtrainable_variables
љregularization_losses
Њ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Ѕ0
¬1
√2
ƒ3
≈4
∆5
«6
»7
…8
 9
Ћ10
ћ11"
trackable_list_wrapper
Q
#0
$1
%2
&3
'4
(5
)6"
trackable_list_wrapper
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ј
“trace_0
”trace_12Е
.__inference_nyan_decoder_layer_call_fn_7408660
.__inference_nyan_decoder_layer_call_fn_7411206Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z“trace_0z”trace_1
ц
‘trace_0
’trace_12ї
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7411232
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408728Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0z’trace_1
“
÷	variables
„trainable_variables
Ўregularization_losses
ў	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses
№
activation

#kernel
$bias"
_tf_keras_layer
Ѕ
Ё	variables
ёtrainable_variables
яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
Ѕ
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
@
й0
к1
л2
м3"
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
№Bў
%__inference_signature_wrapper_7409409input_1input_2input_3"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
н	variables
о	keras_api

пtotal

рcount"
_tf_keras_metric
c
с	variables
т	keras_api

уtotal

фcount
х
_fn_kwargs"
_tf_keras_metric
c
ц	variables
ч	keras_api

шtotal

щcount
ъ
_fn_kwargs"
_tf_keras_metric
R
ы	variables
ь	keras_api

эtotal

юcount"
_tf_keras_metric
R
€	variables
А	keras_api

Бtotal

Вcount"
_tf_keras_metric
'

0"
trackable_list_wrapper
v
?0
@1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
.__inference_nyan_encoder_layer_call_fn_7408015xae"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ХBТ
.__inference_nyan_encoder_layer_call_fn_7410395inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ХBТ
.__inference_nyan_encoder_layer_call_fn_7410457inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
АBэ
.__inference_nyan_encoder_layer_call_fn_7408429xae"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
∞B≠
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7410823inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
∞B≠
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7411189inputs/0inputs/1inputs/2"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЫBШ
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408507xae"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЫBШ
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408585xae"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
е
Иtrace_0
Йtrace_12™
/__inference_graph_masking_layer_call_fn_7411237
/__inference_graph_masking_layer_call_fn_7411242≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 zИtrace_0zЙtrace_1
Ы
Кtrace_0
Лtrace_12а
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411250
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411258≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 zКtrace_0zЛtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
н
Сtrace_02ќ
'__inference_dense_layer_call_fn_7411267Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zСtrace_0
И
Тtrace_02й
B__inference_dense_layer_call_and_return_conditional_losses_7411298Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zТtrace_0
Ђ
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
м
Юtrace_02Ќ
*__inference_ecc_conv_layer_call_fn_7411314Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
З
Яtrace_02и
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7411395Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЯtrace_0
 "
trackable_list_wrapper
Ђ
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses"
_tf_keras_layer
(
¶0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
о
ђtrace_02ѕ
,__inference_ecc_conv_1_layer_call_fn_7411411Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0
Й
≠trace_02к
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7411492Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≠trace_0
 "
trackable_list_wrapper
Ђ
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"
_tf_keras_layer
(
і0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
о
Їtrace_02ѕ
,__inference_ecc_conv_2_layer_call_fn_7411508Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
Й
їtrace_02к
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7411589Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zїtrace_0
 "
trackable_list_wrapper
Ђ
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
(
¬0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
ч
»trace_02Ў
1__inference_global_sum_pool_layer_call_fn_7411594Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
Т
…trace_02у
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7411600Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z…trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
п
ѕtrace_02–
)__inference_dense_1_layer_call_fn_7411609Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѕtrace_0
К
–trace_02л
D__inference_dense_1_layer_call_and_return_conditional_losses_7411620Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z–trace_0
Ђ
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
п
№trace_02–
)__inference_flatten_layer_call_fn_7411625Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z№trace_0
К
Ёtrace_02л
D__inference_flatten_layer_call_and_return_conditional_losses_7411631Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
п
гtrace_02–
)__inference_dense_2_layer_call_fn_7411640Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zгtrace_0
К
дtrace_02л
D__inference_dense_2_layer_call_and_return_conditional_losses_7411651Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
Ђ
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
о
рtrace_02ѕ
(__inference_z_mean_layer_call_fn_7411660Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zрtrace_0
Й
сtrace_02к
C__inference_z_mean_layer_call_and_return_conditional_losses_7411670Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zсtrace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
с
чtrace_02“
+__inference_z_log_var_layer_call_fn_7411679Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zчtrace_0
М
шtrace_02н
F__inference_z_log_var_layer_call_and_return_conditional_losses_7411689Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zшtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
ї	variables
Љtrainable_variables
љregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
р
юtrace_02—
*__inference_sampling_layer_call_fn_7411706Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zюtrace_0
Л
€trace_02м
E__inference_sampling_layer_call_and_return_conditional_losses_7411785Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z€trace_0
I
	?layer
А	variables
Бoutputs"
trackable_dict_wrapper
I
	@layer
В	variables
Гoutputs"
trackable_dict_wrapper
I
	Alayer
Д	variables
Еoutputs"
trackable_dict_wrapper
I
	Blayer
Ж	variables
Зoutputs"
trackable_dict_wrapper
I
	Clayer
И	variables
Йoutputs"
trackable_dict_wrapper
I
	Dlayer
К	variables
Лoutputs"
trackable_dict_wrapper
I
	Elayer
М	variables
Нoutputs"
trackable_dict_wrapper
I
	Flayer
О	variables
Пoutputs"
trackable_dict_wrapper
I
	Glayer
Р	variables
Сoutputs"
trackable_dict_wrapper
I
	Hlayer
Т	variables
Уoutputs"
trackable_dict_wrapper
I
	Ilayer
Ф	variables
Хoutputs"
trackable_dict_wrapper
I
	Jlayer
Ц	variables
Чoutputs"
trackable_dict_wrapper
'
)0"
trackable_list_wrapper
5
R0
S1
T2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
.__inference_nyan_decoder_layer_call_fn_7408660input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
вBя
.__inference_nyan_decoder_layer_call_fn_7411206inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7411232inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408728input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
÷	variables
„trainable_variables
Ўregularization_losses
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
п
Эtrace_02–
)__inference_dense_3_layer_call_fn_7411794Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
К
Юtrace_02л
D__inference_dense_3_layer_call_and_return_conditional_losses_7411805Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
Ђ
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
£__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
Ё	variables
ёtrainable_variables
яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
п
™trace_02–
)__inference_dense_4_layer_call_fn_7411814Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
К
Ђtrace_02л
D__inference_dense_4_layer_call_and_return_conditional_losses_7411825Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
п
±trace_02–
)__inference_dense_5_layer_call_fn_7411834Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z±trace_0
К
≤trace_02л
D__inference_dense_5_layer_call_and_return_conditional_losses_7411844Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0
I
	Rlayer
≥	variables
іoutputs"
trackable_dict_wrapper
I
	Slayer
µ	variables
ґoutputs"
trackable_dict_wrapper
I
	Tlayer
Ј	variables
Єoutputs"
trackable_dict_wrapper
;
єconcat
Їoutputs"
trackable_dict_wrapper
0
п0
р1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
:  (2total
:  (2count
0
у0
ф1"
trackable_list_wrapper
.
с	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ш0
щ1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
э0
ю1"
trackable_list_wrapper
.
ы	variables"
_generic_user_object
+:)  (2vae/nyan_encoder/sampling/total
+:)  (2vae/nyan_encoder/sampling/count
0
Б0
В1"
trackable_list_wrapper
.
€	variables"
_generic_user_object
+:)  (2vae/nyan_encoder/sampling/total
+:)  (2vae/nyan_encoder/sampling/count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBГ
/__inference_graph_masking_layer_call_fn_7411237inputs"≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЖBГ
/__inference_graph_masking_layer_call_fn_7411242inputs"≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
°BЮ
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411250inputs"≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
°BЮ
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411258inputs"≈
Љ≤Є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_dense_layer_call_fn_7411267inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_dense_layer_call_and_return_conditional_losses_7411298inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
0
Б0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBх
*__inference_ecc_conv_layer_call_fn_7411314inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
УBР
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7411395inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
јnon_trainable_variables
Ѕlayers
¬metrics
 √layer_regularization_losses
ƒlayer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѕ
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
0
К0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
,__inference_ecc_conv_1_layer_call_fn_7411411inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7411492inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѕ
–	variables
—trainable_variables
“regularization_losses
”	keras_api
‘__call__
+’&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
0
У0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
,__inference_ecc_conv_2_layer_call_fn_7411508inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7411589inputs/0inputs/1inputs/2mask/0"Ю
Ч≤У
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
÷non_trainable_variables
„layers
Ўmetrics
 ўlayer_regularization_losses
Џlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѕ
џ	variables
№trainable_variables
Ёregularization_losses
ё	keras_api
я__call__
+а&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
1__inference_global_sum_pool_layer_call_fn_7411594inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7411600inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_1_layer_call_fn_7411609inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_1_layer_call_and_return_conditional_losses_7411620inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_flatten_layer_call_fn_7411625inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_flatten_layer_call_and_return_conditional_losses_7411631inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
(
Ѓ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_2_layer_call_fn_7411640inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_2_layer_call_and_return_conditional_losses_7411651inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_z_mean_layer_call_fn_7411660inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_z_mean_layer_call_and_return_conditional_losses_7411670inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_z_log_var_layer_call_fn_7411679inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_z_log_var_layer_call_and_return_conditional_losses_7411689inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^kl_loss
_kl_loss_beta"
trackable_dict_wrapper
тBп
*__inference_sampling_layer_call_fn_7411706inputs/0inputs/1inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
E__inference_sampling_layer_call_and_return_conditional_losses_7411785inputs/0inputs/1inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
№0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_3_layer_call_fn_7411794inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_3_layer_call_and_return_conditional_losses_7411805inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Я	variables
†trainable_variables
°regularization_losses
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_4_layer_call_fn_7411814inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_4_layer_call_and_return_conditional_losses_7411825inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_5_layer_call_fn_7411834inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_5_layer_call_and_return_conditional_losses_7411844inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
–	variables
—trainable_variables
“regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
џ	variables
№trainable_variables
Ёregularization_losses
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
4:22$Adam/vae/nyan_encoder/dense/kernel/m
.:,2"Adam/vae/nyan_encoder/dense/bias/m
<:: 2,Adam/vae/nyan_encoder/ecc_conv/root_kernel/m
1:/ 2%Adam/vae/nyan_encoder/ecc_conv/bias/m
@:>	А2/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/m
::8А2-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/m
>:<  2.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/m
3:1 2'Adam/vae/nyan_encoder/ecc_conv_1/bias/m
B:@	А21Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/m
<::А2/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/m
>:<  2.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/m
3:1 2'Adam/vae/nyan_encoder/ecc_conv_2/bias/m
B:@	А21Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/m
<::А2/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/m
7:5	 А2&Adam/vae/nyan_encoder/dense_1/kernel/m
1:/А2$Adam/vae/nyan_encoder/dense_1/bias/m
8:6
АА2&Adam/vae/nyan_encoder/dense_2/kernel/m
1:/А2$Adam/vae/nyan_encoder/dense_2/bias/m
6:4	А@2%Adam/vae/nyan_encoder/z_mean/kernel/m
/:-@2#Adam/vae/nyan_encoder/z_mean/bias/m
9:7	А@2(Adam/vae/nyan_encoder/z_log_var/kernel/m
2:0@2&Adam/vae/nyan_encoder/z_log_var/bias/m
7:5	@А2&Adam/vae/nyan_decoder/dense_3/kernel/m
1:/А2$Adam/vae/nyan_decoder/dense_3/bias/m
8:6
АІ2&Adam/vae/nyan_decoder/dense_4/kernel/m
1:/І2$Adam/vae/nyan_decoder/dense_4/bias/m
8:6
АЌ2&Adam/vae/nyan_decoder/dense_5/kernel/m
1:/Ќ2$Adam/vae/nyan_decoder/dense_5/bias/m
4:22$Adam/vae/nyan_encoder/dense/kernel/v
.:,2"Adam/vae/nyan_encoder/dense/bias/v
<:: 2,Adam/vae/nyan_encoder/ecc_conv/root_kernel/v
1:/ 2%Adam/vae/nyan_encoder/ecc_conv/bias/v
@:>	А2/Adam/vae/nyan_encoder/ecc_conv/FGN_out/kernel/v
::8А2-Adam/vae/nyan_encoder/ecc_conv/FGN_out/bias/v
>:<  2.Adam/vae/nyan_encoder/ecc_conv_1/root_kernel/v
3:1 2'Adam/vae/nyan_encoder/ecc_conv_1/bias/v
B:@	А21Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/kernel/v
<::А2/Adam/vae/nyan_encoder/ecc_conv_1/FGN_out/bias/v
>:<  2.Adam/vae/nyan_encoder/ecc_conv_2/root_kernel/v
3:1 2'Adam/vae/nyan_encoder/ecc_conv_2/bias/v
B:@	А21Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/kernel/v
<::А2/Adam/vae/nyan_encoder/ecc_conv_2/FGN_out/bias/v
7:5	 А2&Adam/vae/nyan_encoder/dense_1/kernel/v
1:/А2$Adam/vae/nyan_encoder/dense_1/bias/v
8:6
АА2&Adam/vae/nyan_encoder/dense_2/kernel/v
1:/А2$Adam/vae/nyan_encoder/dense_2/bias/v
6:4	А@2%Adam/vae/nyan_encoder/z_mean/kernel/v
/:-@2#Adam/vae/nyan_encoder/z_mean/bias/v
9:7	А@2(Adam/vae/nyan_encoder/z_log_var/kernel/v
2:0@2&Adam/vae/nyan_encoder/z_log_var/bias/v
7:5	@А2&Adam/vae/nyan_decoder/dense_3/kernel/v
1:/А2$Adam/vae/nyan_decoder/dense_3/bias/v
8:6
АІ2&Adam/vae/nyan_decoder/dense_4/kernel/v
1:/І2$Adam/vae/nyan_decoder/dense_4/bias/v
8:6
АЌ2&Adam/vae/nyan_decoder/dense_5/kernel/v
1:/Ќ2$Adam/vae/nyan_decoder/dense_5/bias/vФ
"__inference__wrapped_model_7407447н% !"
эюБВ#$%&'(НҐЙ
БҐ~
|Ґy
%К"
input_1€€€€€€€€€<
%К"
input_2€€€€€€€€€<<	
)К&
input_3€€€€€€€€€<<
™ "4™1
/
output_1#К 
output_1€€€€€€€€€ф•
D__inference_dense_1_layer_call_and_return_conditional_losses_7411620]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
)__inference_dense_1_layer_call_fn_7411609P/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€А¶
D__inference_dense_2_layer_call_and_return_conditional_losses_7411651^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ~
)__inference_dense_2_layer_call_fn_7411640Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А•
D__inference_dense_3_layer_call_and_return_conditional_losses_7411805]#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
)__inference_dense_3_layer_call_fn_7411794P#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€А¶
D__inference_dense_4_layer_call_and_return_conditional_losses_7411825^%&0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€І
Ъ ~
)__inference_dense_4_layer_call_fn_7411814Q%&0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€І¶
D__inference_dense_5_layer_call_and_return_conditional_losses_7411844^'(0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€Ќ
Ъ ~
)__inference_dense_5_layer_call_fn_7411834Q'(0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ќ™
B__inference_dense_layer_call_and_return_conditional_losses_7411298d3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€<
™ ")Ґ&
К
0€€€€€€€€€<
Ъ В
'__inference_dense_layer_call_fn_7411267W3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€<
™ "К€€€€€€€€€<ћ
G__inference_ecc_conv_1_layer_call_and_return_conditional_losses_7411492АћҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€< 
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 ")Ґ&
К
0€€€€€€€€€< 
Ъ §
,__inference_ecc_conv_1_layer_call_fn_7411411ућҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€< 
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 "К€€€€€€€€€< ћ
G__inference_ecc_conv_2_layer_call_and_return_conditional_losses_7411589АћҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€< 
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 ")Ґ&
К
0€€€€€€€€€< 
Ъ §
,__inference_ecc_conv_2_layer_call_fn_7411508ућҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€< 
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 "К€€€€€€€€€<  
E__inference_ecc_conv_layer_call_and_return_conditional_losses_7411395АћҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 ")Ґ&
К
0€€€€€€€€€< 
Ъ Ґ
*__inference_ecc_conv_layer_call_fn_7411314ућҐ»
ЕҐБ
Ъ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
>™;
9
mask1Ъ.
$К!
mask/0€€€€€€€€€<

 

 "К€€€€€€€€€< Ґ
D__inference_flatten_layer_call_and_return_conditional_losses_7411631Z0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ z
)__inference_flatten_layer_call_fn_7411625M0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ађ
L__inference_global_sum_pool_layer_call_and_return_conditional_losses_7411600\3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€< 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ Д
1__inference_global_sum_pool_layer_call_fn_7411594O3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€< 
™ "К€€€€€€€€€ Њ
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411250pCҐ@
)Ґ&
$К!
inputs€€€€€€€€€<
™

trainingp ")Ґ&
К
0€€€€€€€€€<
Ъ Њ
J__inference_graph_masking_layer_call_and_return_conditional_losses_7411258pCҐ@
)Ґ&
$К!
inputs€€€€€€€€€<
™

trainingp")Ґ&
К
0€€€€€€€€€<
Ъ Ц
/__inference_graph_masking_layer_call_fn_7411237cCҐ@
)Ґ&
$К!
inputs€€€€€€€€€<
™

trainingp "К€€€€€€€€€<Ц
/__inference_graph_masking_layer_call_fn_7411242cCҐ@
)Ґ&
$К!
inputs€€€€€€€€€<
™

trainingp"К€€€€€€€€€<ѓ
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7408728b#$%&'(0Ґ-
&Ґ#
!К
input_1€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ Ѓ
I__inference_nyan_decoder_layer_call_and_return_conditional_losses_7411232a#$%&'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ З
.__inference_nyan_decoder_layer_call_fn_7408660U#$%&'(0Ґ-
&Ґ#
!К
input_1€€€€€€€€€@
™ "К€€€€€€€€€фЖ
.__inference_nyan_decoder_layer_call_fn_7411206T#$%&'(/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ф±
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408507г !"
эюБВКҐЖ
oҐl
jҐg
К
x€€€€€€€€€<
К
a€€€€€€€€€<<	
#К 
e€€€€€€€€€<<
™

trainingp "3Ґ0
К
0€€€€€€€€€@
Ъ
К	
1/0 ±
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7408585г !"
эюБВКҐЖ
oҐl
jҐg
К
x€€€€€€€€€<
К
a€€€€€€€€€<<	
#К 
e€€€€€€€€€<<
™

trainingp"3Ґ0
К
0€€€€€€€€€@
Ъ
К	
1/0 »
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7410823ъ !"
эюБВ°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp "3Ґ0
К
0€€€€€€€€€@
Ъ
К	
1/0 »
I__inference_nyan_encoder_layer_call_and_return_conditional_losses_7411189ъ !"
эюБВ°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp"3Ґ0
К
0€€€€€€€€€@
Ъ
К	
1/0 ы
.__inference_nyan_encoder_layer_call_fn_7408015» !"
эюБВКҐЖ
oҐl
jҐg
К
x€€€€€€€€€<
К
a€€€€€€€€€<<	
#К 
e€€€€€€€€€<<
™

trainingp "К€€€€€€€€€@ы
.__inference_nyan_encoder_layer_call_fn_7408429» !"
эюБВКҐЖ
oҐl
jҐg
К
x€€€€€€€€€<
К
a€€€€€€€€€<<	
#К 
e€€€€€€€€€<<
™

trainingp"К€€€€€€€€€@Т
.__inference_nyan_encoder_layer_call_fn_7410395я !"
эюБВ°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp "К€€€€€€€€€@Т
.__inference_nyan_encoder_layer_call_fn_7410457я !"
эюБВ°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp"К€€€€€€€€€@У
E__inference_sampling_layer_call_and_return_conditional_losses_7411785…эюБВЕҐБ
zҐw
uЪr
"К
inputs/0€€€€€€€€€@
"К
inputs/1€€€€€€€€€@
(Т%	Ґ
ъ 
А
p VariableSpec 
™ "5Ґ2
К
0€€€€€€€€€@
Ъ
К
1/0џ
*__inference_sampling_layer_call_fn_7411706ђэюБВЕҐБ
zҐw
uЪr
"К
inputs/0€€€€€€€€€@
"К
inputs/1€€€€€€€€€@
(Т%	Ґ
ъ 
А
p VariableSpec 
™ "К€€€€€€€€€@і
%__inference_signature_wrapper_7409409К% !"
эюБВ#$%&'(™Ґ¶
Ґ 
Ю™Ъ
0
input_1%К"
input_1€€€€€€€€€<
0
input_2%К"
input_2€€€€€€€€€<<	
4
input_3)К&
input_3€€€€€€€€€<<"4™1
/
output_1#К 
output_1€€€€€€€€€ф¬
@__inference_vae_layer_call_and_return_conditional_losses_7409254э% !"
эюБВ#$%&'(ЭҐЩ
БҐ~
|Ґy
%К"
input_1€€€€€€€€€<
%К"
input_2€€€€€€€€€<<	
)К&
input_3€€€€€€€€€<<
™

trainingp "4Ґ1
К
0€€€€€€€€€ф
Ъ
К	
1/0 ¬
@__inference_vae_layer_call_and_return_conditional_losses_7409330э% !"
эюБВ#$%&'(ЭҐЩ
БҐ~
|Ґy
%К"
input_1€€€€€€€€€<
%К"
input_2€€€€€€€€€<<	
)К&
input_3€€€€€€€€€<<
™

trainingp"4Ґ1
К
0€€€€€€€€€ф
Ъ
К	
1/0 ∆
@__inference_vae_layer_call_and_return_conditional_losses_7409945Б% !"
эюБВ#$%&'(°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp "4Ґ1
К
0€€€€€€€€€ф
Ъ
К	
1/0 ∆
@__inference_vae_layer_call_and_return_conditional_losses_7410333Б% !"
эюБВ#$%&'(°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp"4Ґ1
К
0€€€€€€€€€ф
Ъ
К	
1/0 М
%__inference_vae_layer_call_fn_7408880в% !"
эюБВ#$%&'(ЭҐЩ
БҐ~
|Ґy
%К"
input_1€€€€€€€€€<
%К"
input_2€€€€€€€€€<<	
)К&
input_3€€€€€€€€€<<
™

trainingp "К€€€€€€€€€фМ
%__inference_vae_layer_call_fn_7409178в% !"
эюБВ#$%&'(ЭҐЩ
БҐ~
|Ґy
%К"
input_1€€€€€€€€€<
%К"
input_2€€€€€€€€€<<	
)К&
input_3€€€€€€€€€<<
™

trainingp"К€€€€€€€€€фР
%__inference_vae_layer_call_fn_7409483ж% !"
эюБВ#$%&'(°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp "К€€€€€€€€€фР
%__inference_vae_layer_call_fn_7409557ж% !"
эюБВ#$%&'(°ҐЭ
ЕҐБ
Ґ|
&К#
inputs/0€€€€€€€€€<
&К#
inputs/1€€€€€€€€€<<	
*К'
inputs/2€€€€€€€€€<<
™

trainingp"К€€€€€€€€€фІ
F__inference_z_log_var_layer_call_and_return_conditional_losses_7411689]!"0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ 
+__inference_z_log_var_layer_call_fn_7411679P!"0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@§
C__inference_z_mean_layer_call_and_return_conditional_losses_7411670] 0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
(__inference_z_mean_layer_call_fn_7411660P 0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@