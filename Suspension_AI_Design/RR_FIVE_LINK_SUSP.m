clear;
clc;
global A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 C C1 LUP LLWR SCLP 

% 脚本功能总览：
% 1) 以悬架硬点为输入，按轮心 Z 方向行程逐步扫描（Rebound -> Bounce）。
% 2) 每个行程点通过 fsolve 求解刚体姿态参数，使转向节/轮心满足几何约束。
% 3) 由姿态矩阵计算各外点坐标，并修正连杆长度不变约束。
% 4) 计算 camber、toe、轮胎接地点轨迹。
% 5) 基于接地点轨迹三点等距法求侧倾中心高度（RCH）随轮跳变化。

%% 初始化
BOUstroke=100;%定义轮胎上跳行程
REBstroke=-100;%定义轮胎下跳行程
tire_radius=352;%定义轮胎静力半径
step=5;%定义轮跳步长

%% 硬点定义
A1=[4313,-428,1054.5];%上前控制臂内点
B1=[4365.008,-715.386,1065.326];%上前控制臂外点
A2=[4588,-357,1090.5];%上后控制臂内点
B2=[4444.123,-697.692,1111.512];%上后控制臂外点
A3=[4242,-413,863.5];%LWR LINK FR INR
B3=[4391.608,-730.852,801.842];%LWR LINK FR OTR
A4=[4709.907,-292.376,865.386];%LWR LINK RR INR
B4=[4535.801,-714.774,889.46];%LWR LINK RR OTG
A5=[4316,-446,936.5];%ROD INR
B5=[4292.353,-722.178,943.276];%ROD OTR 
C=[4421.553,-825.1,972.284];%wheel_center
C1=[4421.811,-745.143,970.287];%%KNU REF
LUP=[4630.495,-640.275,1044.481];%STAB LINK UP
LLWR=[4617.547,-667.336,882.149];%STAB LINK LWR
SCLP=[4881,-480,1002.5];%STAB CLAMP

%% 初始化循环参数
k=0;%定义循环参数
h=0;%定义循环参数，瞬时螺旋轴
Var_WCTR=0;%定义循环参数，侧倾中心
CONT_PONT=[];
CP_anti=[];
ff=optimset;
ff.TolX=1e-15;ff.TolFun=1e-20;ff.Display='iter';
% fsolve 收敛设置：
% TolX/TolFun 设置较小，追求几何求解的高精度；Display='iter' 便于观察迭代过程。

%% 
UPRBOU=BOUstroke+C(3);
LWRREB=REBstroke+C(3);
last_x=[];
last_y=[];
last_exitflag=[];
last_Ctz=[];
for Ctz=LWRREB:step:UPRBOU
    % Ctz: 当前步的轮心目标 Z 坐标（绝对坐标），从下跳扫到上跳。
    
    [x,y,exitflag]=fsolve(@(a)XP_algorithm(a,Ctz),[0;0;0;4421.553;-825.1],ff);%%a,b,r,Ctx,Cty
    last_x=x;
    last_y=y;
    last_exitflag=exitflag;
    last_Ctz=Ctz;
    % x(1:3): 姿态角（用于构造旋转矩阵 E）
    % x(4:5): 轮心平面坐标 Ctx/Cty
    % Ctz   : 轮心竖向坐标（本循环已给定）
    % 这里把“轮心目标高度 + 连杆约束”统一交给 XP_algorithm 非线性方程求解。

    % 由姿态角构造 3x3 旋转矩阵 E。
    d11=cos(x(1))*cos(x(2));
    d12=-sin(x(1))*cos(x(2));
    d13=sin(x(2));
    d21=sin(x(1))*cos(x(3))+cos(x(1))*sin(x(2))*sin(x(3));
    d22=cos(x(1))*cos(x(3))-sin(x(1))*sin(x(2))*sin(x(3));
    d23=cos(x(2))*sin(x(3));
    d31=sin(x(1))*sin(x(3))-cos(x(1))*sin(x(2))*cos(x(3));
    d32=cos(x(1))*sin(x(3))+sin(x(1))*sin(x(2))*cos(x(3));
    d33=cos(x(2))*cos(x(3));
    E=[d11,d12,d13;d21,d22,d23;d31,d32,d33];
    % 平移向量 Y 的构造方式：保证变换后轮心 C 的坐标恰好等于 [Ctx,Cty,Ctz]。
    Y=[x(4)-(d11*C(1)+d12*C(2)+d13*C(3));x(5)-(d21*C(1)+d22*C(2)+d23*C(3));Ctz-(d31*C(1)+d32*C(2)+d33*C(3))];
    DOt=[E,Y;0,0,0,1];
    % DOt: 4x4 齐次变换矩阵。用于把初始几何点统一变换到当前姿态。

    B11t=DOt*[B1,1]';
    B22t=DOt*[B2,1]';
    B33t=DOt*[B3,1]';
    B44t=DOt*[B4,1]';
    B55t=DOt*[B5,1]';
    C11t=DOt*[C1,1]';
    LUPt=DOt*[LUP,1]';

    % 变换后外点可能会破坏“连杆长度恒定”约束。
    % 这里把每个外点的 y 分量设为未知量，反解使 |Ai-Bit| 等于初始长度 |Ai-Bi|。
    syms B11y B22y B33y B44y B55y;
    FunB11y=sqrt((B11t(1)-A1(1))^2+(B11y-A1(2))^2+(B11t(3)-A1(3))^2)-norm(B1-A1);
    FunB22y=sqrt((B22t(1)-A2(1))^2+(B22y-A2(2))^2+(B22t(3)-A2(3))^2)-norm(B2-A2);
    FunB33y=sqrt((B33t(1)-A3(1))^2+(B33y-A3(2))^2+(B33t(3)-A3(3))^2)-norm(B3-A3);
    FunB44y=sqrt((B44t(1)-A4(1))^2+(B44y-A4(2))^2+(B44t(3)-A4(3))^2)-norm(B4-A4);
    FunB55y=sqrt((B55t(1)-A5(1))^2+(B55y-A5(2))^2+(B55t(3)-A5(3))^2)-norm(B5-A5);
    [B11y1, B22y1, B33y1, B44y1, B55y1]=solve(FunB11y, FunB22y, FunB33y , FunB44y , FunB55y ,B11y, B22y, B33y, B44y, B55y);

    % 每个 y 分量通常有两个数学解，这里统一取较小值（min）作为几何分支选择。
    % 这相当于固定机构装配支路，避免在扫描过程中跳分支。
    BB1(1,1)=double(B11y1 (1));
    BB1(1,2)=double(B11y1 (2));
    TRANS=min(BB1(1,1),BB1(1,2));
    B11t(2)=TRANS;

    BB1(1,1)=double(B22y1 (1));
    BB1(1,2)=double(B22y1 (2));
    TRANS=min(BB1(1,1),BB1(1,2));
    B22t(2)=TRANS;

    BB1(1,1)=double(B33y1(1));
    BB1(1,2)=double(B33y1 (2));
    TRANS =min(BB1(1,1),BB1(1,2));
    B33t(2)=TRANS;

    BB1 (1,1)=double(B44y1 (1));
    BB1(1,2)=double(B44y1 (2));
    TRANS =min(BB1(1,1),BB1(1,2));
    B44t(2)= TRANS;

    BB1 (1,1)=double(B55y1 (1));
    BB1(1,2)=double(B55y1 (2));
    TRANS =min(BB1(1,1),BB1(1,2));
    B55t(2)= TRANS;

    Blt=B11t(1:3,:);
    B2t=B22t(1:3,:);
    B3t=B33t(1:3,:);
    B4t=B44t(1:3,:);
    B5t=B55t(1:3,:);
    Ct=[x(4),x(5),Ctz];%轮心
    C1t=C11t(1:3,:);%轮心方向点
    
    k=k+1;
    Ctz_plot(k)=Ctz-C(3);%轮跳量输出
    Ct_write(k,1)=Ct(1);%轮心数据输出
    Ct_write(k,2)=Ct(2);%轮心数据输出
    Ct_write(k,3)=Ct(3);%轮心数据输出
    
    KNU_write(k,1)=C1t(1);%KNU数据输出
    KNU_write(k,2)=C1t(2);%KNU数据输出
    KNU_write(k,3)=C1t(3);%KNU数据输出
    
    Blt_write(k,1)=Blt(1);
    Blt_write(k,2)=Blt(2);
    Blt_write(k,3)=Blt(3);
    
    B2t_write(k,1)=B2t(1);
    B2t_write(k,2)=B2t(2);
    B2t_write(k,3)=B2t(3);
    
    B3t_write(k,1)=B3t(1);
    B3t_write(k,2)=B3t(2);
    B3t_write(k,3)=B3t(3);
    
    B4t_write(k,1)=B4t(1);
    B4t_write(k,2)=B4t(2);
    B4t_write(k,3)=B4t(3); 
    
    B5t_write(k,1)=B5t(1);
    B5t_write(k,2)=B5t(2);
    B5t_write(k,3)=B5t(3);
    
    delta_z=Ctz-C1t(3);
    delta_y=Ct(2)-C1t(2);
    delta_x=Ct(1)-C1t(1);
    % camber/toe 由“轮心到方向点”的相对方向计算，等效于车轮局部姿态角。
    camber(k)=atan(delta_z/delta_y)*180/pi;%输出camber angle
    toe(k)=atan(delta_x/delta_y)*180/pi;%输出toe angle
    
%% 接地点坐标
    % 目标：求轮胎与地面接地点 CONT_PONT。
    % 思路：
    % 1) 以轮心 Ct 为球心、tire_radius 为半径建立球面方程。
    % 2) 加上“接地点矢量与轮心轴向矢量正交”约束，联立求 Jx/Jy。
    % 3) 通过参数 t 在 Ct -> TT_CTR 连线上精化到球面上，得到最终接地点。
    syms Jx Jy Jz CONTx CONTy CONTz t
    Jz=Ctz-10;
    Var_WCTR=Var_WCTR+1;
    Ct=Ct';
    WHEEL=C1t-Ct;
    WHEEL1=WHEEL(1:3,:);
    TIRE=[Jx-Ct(1),Jy-Ct(2),Jz-Ct(3)];
    TIRE_RA1=sqrt((Jx-Ct(1))^2+(Jy-Ct(2))^2+(Jz-Ct(3))^2)-tire_radius;
    TIRE_RA2=WHEEL1(1)*TIRE(1)+WHEEL1(2)*TIRE(2)+WHEEL1(3)*TIRE(3);
    [Jx1,Jy1]=solve(TIRE_RA1,TIRE_RA2,Jx,Jy);
    TT1(Var_WCTR,1)=double(Jx1(1));
    TT1(Var_WCTR,2)=double(Jx1(2));
    TT2(Var_WCTR,1)=double(Jy1(1));
    TT2(Var_WCTR,2)=double(Jy1(2));
    TT_CTR=[(TT1(Var_WCTR,1)+TT1(Var_WCTR,2))/2,(TT2(Var_WCTR,1)+TT2(Var_WCTR,2))/2,Jz];
    Ct1=Ct(1:3,:)';
    CONTx=(TT_CTR(1)-Ct1(1))*t+Ct1(1);
    CONTy=(TT_CTR(2)-Ct1(2))*t+Ct1(2);
    CONTz=(TT_CTR(3)-Ct1(3))*t+Ct1(3);
    CONT=[CONTx,CONTy,CONTz];
    TIRE_RA3=norm(CONT-Ct1)-tire_radius;
    [xxx]=solve(TIRE_RA3,t);
    t=double(xxx);
    CONTx=(TT_CTR(1)-Ct1(1))*t+Ct1(1);
    CONTy=(TT_CTR(2)-Ct1(2))*t+Ct1(2);
    CONTz=(TT_CTR(3)-Ct1(3))*t+Ct1(3);
    CONT_PONT=[CONT_PONT;CONTx,CONTy,CONTz];%输出轮胎接地点坐标
end

%% 侧倾中心高
syms y0 z0
num=(UPRBOU-LWRREB)/step+1;
aa=40/step;
ab=[];
RCH=[];
ycc=[];zcc=[];yaa=[];zaa=[];ybb=[];zbb=[];WC_RCH=[];
for i=1:(num-2*aa)
    % 三点法：在接地点轨迹上取 (i, i+aa, i+2aa) 三个点，
    % 构造到假想旋转中心 (y0,z0) 的等距条件，反求该瞬时中心位置。
    % 最终把中心投影到车辆中心面得到 h，作为该工况下侧倾中心高度。
    ab=[ab;i];
    yc=CONT_PONT(i,2);
    zc=CONT_PONT(i,3);
    ya=CONT_PONT(i+aa,2);
    za=CONT_PONT(i+aa,3);
    yb=CONT_PONT(i+2*aa,2);
    zb=CONT_PONT(i+2*aa,3);
    ycc=[ycc;yc];
    zcc=[zcc;zc];
    yaa=[yaa;ya];
    zaa=[zaa;za];
    ybb=[ybb;yb];
    zbb=[zbb;zb];
    sd1=sqrt((y0-yc)^2+(z0-zc)^2)-sqrt((y0-ya)^2+(z0-za)^2);
    sd2=sqrt((y0-ya)^2+(z0-za)^2)-sqrt((y0-yb)^2+(z0-zb)^2);
    [yy1,yy2]=solve(sd1,sd2,y0,z0);
    y00=double(yy1);
    z00=double(yy2);
    h=ya*(z00-za)/(ya-y00);
    RCH=[RCH;h]; %输出侧倾中心高
    WC_RCH=[WC_RCH;Ctz_plot(i+aa)];%输出轮跳
end

%% 打印最后一次优化参数
fprintf('\nFinal optimization result (MATLAB):\n');
fprintf('Ctz = %.6f\n', last_Ctz);
fprintf('exitflag = %d\n', last_exitflag);
fprintf('x = [%.12f, %.12f, %.12f, %.12f, %.12f]\n', last_x(1), last_x(2), last_x(3), last_x(4), last_x(5));
fprintf('residual y = [%.12e, %.12e, %.12e, %.12e, %.12e]\n', last_y(1), last_y(2), last_y(3), last_y(4), last_y(5));
