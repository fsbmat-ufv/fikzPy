; Offline Windows x64 installer. Build through scripts/build_windows_installer.py.
#define MyAppName "fikzPy"
#ifndef MyAppVersion
  #define MyAppVersion "0.1.1"
#endif

[Setup]
#ifdef ValidationBuild
AppId=fikzPy-Installer-Validation
#else
AppId={{E7B8C4D2-5A91-4F63-B0D8-2C6F1A9E4B37}
#endif
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Fernando de Souza Bastos
AppContact=fernando.bastos@ufv.br
AppSupportURL=https://github.com/fsbmat-ufv/fikzPy
AppUpdatesURL=https://github.com/fsbmat-ufv/fikzPy/releases
DefaultDirName={localappdata}\Programs\fikzPy
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
InfoBeforeFile=info_criacao.txt
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=fikzPy-Setup-{#MyAppVersion}-Windows-x64
#ifdef ValidationBuild
Compression=none
SolidCompression=no
#else
Compression=lzma2/fast
SolidCompression=yes
#endif
WizardStyle=modern
UninstallDisplayIcon={app}\fikzPy.exe
CloseApplications=yes

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
brazilianportuguese.WelcomeLabel2=Este assistente instala o fikzPy completo: aplicativo, Python, bibliotecas e LaTeX para gerar PDFs.%n%nNão é necessário instalar dependências separadamente nem conectar à internet.%n%nAutor: Fernando de Souza Bastos (fernando.bastos@ufv.br).
english.WelcomeLabel2=This wizard installs fikzPy, Python, its libraries and a private LaTeX distribution for PDF generation.%n%nNo separate dependencies or internet connection are required.%n%nAuthor: Fernando de Souza Bastos (fernando.bastos@ufv.br).

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\fikzPy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "info_criacao.txt"; DestDir: "{app}"; DestName: "SOBRE.txt"; Flags: ignoreversion

#ifndef ValidationBuild
[Icons]
Name: "{autoprograms}\fikzPy"; Filename: "{app}\fikzPy.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\fikzPy"; Filename: "{app}\fikzPy.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autoprograms}\fikzPy - Desinstalar"; Filename: "{uninstallexe}"
#endif

[Run]
Filename: "{app}\fikzPy.exe"; WorkingDir: "{app}"; Description: "Executar o fikzPy agora"; Flags: postinstall nowait skipifsilent
