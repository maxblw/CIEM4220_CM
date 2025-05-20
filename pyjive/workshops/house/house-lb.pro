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

  models = [ frame, diri, neum ];

  frame =
  {
    type = Frame;

    elements = all;

    subtype = linear;

    EA = 10.e6;
    GAs = 10.e6;
    EI = 10.e3;
    Mp = 3.e3;

    shape =
    {
      type = Line2;
      intScheme = Gauss1;
    };
  };

  diri =
  {
    type = Dirichlet; 

    groups = [ bot, bot, bot ];
    dofs   = [ dx, dy, phi ];
    values = [ 0.0, 0.0, 0.0 ];
  };

  neum = 
  {
    type = Neumann;
    groups = [ top ];
    dofs = [ dy ];
    values = [ -1. ];
  };
};

linbuck =
{
  type = LinBuck;
};

frameview =
{
  type = FrameView;
  deform = 1.;
  interactive = True;
};

loaddisp = 
{
  type = LoadDisp;
  groups = [ top ];
};
