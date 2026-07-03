; -----------------------------------------------------------------------------
; Instalador do fikzPy (Inno Setup 6)
;
; Autor do software: Fernando de Souza Bastos <fernando.bastos@ufv.br>
; O fikzPy foi criado com o objetivo de auxiliar na criacao de imagens TikZ
; e na criacao de atividades educacionais.
;
; Para gerar o instalador: compile este arquivo com o ISCC.exe (Inno Setup 6).
; O executavel resultante fica em installer\Output\fikzPy-Setup-<versao>.exe
; -----------------------------------------------------------------------------

#define MyAppName "fikzPy"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Fernando de Souza Bastos"
#define MyAppEmail "fernando.bastos@ufv.br"
#define MyAppURL "https://github.com/fsbmat-ufv/fikzPy"
#define PythonURL "https://www.python.org/ftp/python/3.13.7/python-3.13.7-amd64.exe"

[Setup]
AppId={{E7B8C4D2-5A91-4F63-B0D8-2C6F1A9E4B37}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppContact={#MyAppEmail}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
InfoBeforeFile=info_criacao.txt
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=fikzPy-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
brazilianportuguese.WelcomeLabel2=Este assistente vai instalar o [name/ver] no seu computador.%n%nO fikzPy foi criado por Fernando de Souza Bastos (fernando.bastos@ufv.br) com o objetivo de auxiliar na criação de imagens TikZ e na criação de atividades educacionais.%n%nÉ recomendável fechar todos os outros aplicativos antes de continuar.
english.WelcomeLabel2=This wizard will install [name/ver] on your computer.%n%nfikzPy was created by Fernando de Souza Bastos (fernando.bastos@ufv.br) to help with the creation of TikZ images and educational activities.%n%nIt is recommended that you close all other applications before continuing.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Somente os modulos que o aplicativo usa em tempo de execucao
; (a pasta fikzpy do repositorio contem artefatos de desenvolvimento que nao devem ser distribuidos)
Source: "..\fikzpy\__init__.py"; DestDir: "{app}\fikzpy"; Flags: ignoreversion
Source: "..\fikzpy\main.py"; DestDir: "{app}\fikzpy"; Flags: ignoreversion
Source: "..\fikzpy\core\*"; DestDir: "{app}\fikzpy\core"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\fikzpy\gui\*"; DestDir: "{app}\fikzpy\gui"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\fikzpy\templates\*"; DestDir: "{app}\fikzpy\templates"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements-runtime.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "instalar_dependencias.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "fikzPy.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "info_criacao.txt"; DestDir: "{app}"; DestName: "SOBRE.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m fikzpy.main"; WorkingDir: "{app}"; Comment: "fikzPy — imagens TikZ e atividades educacionais"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m fikzpy.main"; WorkingDir: "{app}"; Comment: "fikzPy — imagens TikZ e atividades educacionais"; Tasks: desktopicon

[Run]
Filename: "{app}\instalar_dependencias.bat"; StatusMsg: "Instalando as dependências Python (isso pode demorar alguns minutos)..."; Flags: waituntilterminated
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m fikzpy.main"; WorkingDir: "{app}"; Description: "Executar o fikzPy agora"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\fikzpy"

[Code]
var
  DownloadPage: TDownloadWizardPage;

{ Executa "python -c ..." e confere se a versao e >= 3.10 }
function PythonCmdOk(const Command: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe',
    '/c ' + Command + ' -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function PythonExeOk(const ExePath: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := FileExists(ExePath) and
    Exec(ExePath, '-c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

{ Procura o Python 3.10+ no PATH, no launcher "py" e nas instalacoes por usuario }
function HasPython(): Boolean;
var
  FindRec: TFindRec;
  Base: String;
begin
  Result := PythonCmdOk('py -3') or PythonCmdOk('python');
  if Result then
    Exit;
  Base := ExpandConstant('{localappdata}\Programs\Python\');
  if FindFirst(Base + 'Python3*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0) and
           PythonExeOk(Base + FindRec.Name + '\python.exe') then
        begin
          Result := True;
          Exit;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing),
    SetupMessage(msgPreparingDesc), nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;
  if HasPython() then
    Exit;

  case SuppressibleMsgBox(
    'O fikzPy precisa do Python 3.10 ou superior, que não foi encontrado neste computador.' + #13#10 + #13#10 +
    'Deseja que o instalador baixe e instale o Python 3.13 automaticamente agora?' + #13#10 + #13#10 +
    'Sim — baixar e instalar o Python automaticamente (requer internet).' + #13#10 +
    'Não — abrir a página de download do Python para instalá-lo manualmente antes de continuar.',
    mbConfirmation, MB_YESNOCANCEL, IDYES) of
    IDYES:
      begin
        DownloadPage.Clear;
        DownloadPage.Add('{#PythonURL}', 'python-setup.exe', '');
        DownloadPage.Show;
        try
          try
            DownloadPage.Download;
          except
            SuppressibleMsgBox(
              'Falha ao baixar o instalador do Python. Verifique sua conexão com a internet, ' +
              'ou instale o Python 3.10+ manualmente (https://www.python.org/downloads/) ' +
              'e execute este instalador novamente.',
              mbError, MB_OK, IDOK);
            Result := False;
            Exit;
          end;
        finally
          DownloadPage.Hide;
        end;
        if not (Exec(ExpandConstant('{tmp}\python-setup.exe'),
                 '/passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0',
                 '', SW_SHOW, ewWaitUntilTerminated, ResultCode)
                and (ResultCode = 0) and HasPython()) then
        begin
          SuppressibleMsgBox(
            'Não foi possível instalar o Python automaticamente. ' +
            'Instale o Python 3.10 ou superior (https://www.python.org/downloads/) ' +
            'e execute este instalador novamente.',
            mbError, MB_OK, IDOK);
          Result := False;
        end;
      end;
    IDNO:
      begin
        ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW, ewNoWait, ResultCode);
        SuppressibleMsgBox(
          'O Python deve ser instalado antes de continuar.' + #13#10 + #13#10 +
          'Instale o Python 3.10 ou superior (marcando a opção "Add python.exe to PATH") ' +
          'e depois execute este instalador novamente.',
          mbInformation, MB_OK, IDOK);
        Result := False;
      end;
  else
    Result := False;
  end;
end;
