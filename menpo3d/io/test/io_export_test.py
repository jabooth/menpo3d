from mock import patch, PropertyMock, MagicMock
import menpo3d.io as mio


test_obj = mio.import_builtin_asset('james.obj')
test_lg = mio.import_landmark_file(mio.data_path_to('bunny.ljson'))


@patch('menpo3d.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_mesh_obj(mock_open, exists):
    exists.return_value = False
    fake_path = '/fake/fake.obj'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_mesh(f, test_obj, extension='obj')


@patch('menpo.image.base.PILImage')
@patch('menpo3d.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_mesh_obj_textured(mock_open, exists, PILImage):
    exists.return_value = False
    mock_open.return_value = MagicMock()
    fake_path = '/fake/fake.obj'
    mio.export_textured_mesh(fake_path, test_obj, extension='obj')
    PILImage.save.assert_called_once()


@patch('menpo.io.output.landmark.json.dump')
@patch('menpo3d.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson(mock_open, exists, json_dump):
    exists.return_value = False
    fake_path = '/fake/fake.ljson'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(f, test_lg, extension='ljson')
    json_dump.assert_called_once()
