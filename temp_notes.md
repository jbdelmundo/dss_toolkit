
==================================================================
Create an animated 3D scatter plot



%matplotlib notebook



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



from IPython.display import HTML
from matplotlib import animation



fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111, projection='3d')



x = df["field1"]
y = df["field2"]
z = df["field3"]
color = df["cluster"]



ax.scatter(x , # X Axis Values
y , # Y Axis Values
z , # Z Axis Values
c=color , # Color of each record
marker='o' , # Marker Style = https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
s=80 , # Marker Size
alpha= .5 , # Blending Value, 0 transparent, 1 opaque
cmap="RdBu"
)



ax.set_xlabel('Final Income')
ax.set_ylabel('Outstanding Balance')
ax.set_zlabel('Max Loan Term')



def animate(frame):
ax.view_init(30, frame/2)
plt.pause(.01)
return fig



anim = animation.FuncAnimation(fig, animate, frames=240, interval=60)


==================================================================


Enable code folding in jupyter notebook



# Install Jupyter Notebook Extensions: Collapse Markdown Cells (# - ######)

!pip install jupyter_contrib_nbextensions
!jupyter contrib nbextension install --user



!jupyter nbextension enable codefolding/main # Enable codefolding
#!jupyter nbextension disable codefolding/main # Disable codefolding



!jupyter nbextension enable collapsible_headings/main # Enable codefolding
#!jupyter nbextension disable collapsible_headings/main # Disable codefolding




### Add extra tab in notebook for managing all installed extensions

!pip install jupyter_nbextensions_configurator
!jupyter nbextensions_configurator enable --user


==================================================================

Display all restuls from jupyter notebook cell# Display full code outputs in cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


==================================================================
Maximize cell width in jupyter notebook



# Script to adjust jupyter notebook cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container {width: 100% !important; }</style>"))