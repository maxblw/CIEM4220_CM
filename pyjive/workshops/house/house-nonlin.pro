init =
{
  nodeGroups = [ bot, top ];

  mesh = 
  {
    type = geo;
    file = house.geom;
  };

  bot = [0, 4];
  top = [2];
};

model =
{
  type = Multi;

  models = [ frame, diri ];

  frame =
  {
    type = Frame;

    elements = all;

    subtype = nonlin;

    EA = 10.e6;
    GAs = 10.e6;
    EI = 10.e3;

    shape =
    {
      type = Line2;
      intScheme = Gauss1;
    };
  };

  diri =
  {
    type = Dirichlet; 

    groups = [ bot, bot, bot, top ];
    dofs   = [ dx, dy, phi, dy];
    values = [ 0.0, 0.0, 0.0, -0.003 ];
    dispIncr = [ 0.0, 0.0, 0.0, -0.003 ];
  };
};

nonlin =
{
  type = Nonlin;
  nsteps = 200;
};

frameview =
{
  type = FrameView;
  deform = 1.;
  interactive = True;
  plotStress = M;
};

loaddisp = 
{
  type = LoadDisp;
  groups = [ top ];
};

