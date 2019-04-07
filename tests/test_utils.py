import pytest
import torch

from maml.modules import MetaLinear
from maml.utils import update_parameters

@pytest.fixture
def model():
    model = MetaLinear(3, 1, bias=False)
    model.weight.data = torch.tensor([[2., 3., 5.]])
    return model

def test_update_parameters(model):
    """
    The loss function (with respect to the weights of the model w) is defined as
        f(w) = 0.5 * (7 * w_1 + 11 * w_2 + 13 * w_3) ** 2
    with w = [2, 3, 5].

    The gradient of the function f with respect to w, and evaluated
    on w = [2, 3, 5], is:
        df / dw_1 =  7 * (7 * w_1 + 11 * w_2 + 13 * w_3) =  784
        df / dw_2 = 11 * (7 * w_1 + 11 * w_2 + 13 * w_3) = 1232
        df / dw_3 = 13 * (7 * w_1 + 11 * w_2 + 13 * w_3) = 1222

    The updated parameter w' is then given by one step of gradient descent,
    with step size 0.5:
        w'_1 = w_1 - 0.5 * df / dw_1 = 2 - 0.5 *  784 = -390
        w'_2 = w_2 - 0.5 * df / dw_2 = 3 - 0.5 * 1232 = -613
        w'_3 = w_3 - 0.5 * df / dw_3 = 5 - 0.5 * 1456 = -723
    """
    train_inputs = torch.tensor([[7., 11., 13.]])
    train_loss = 0.5 * (model(train_inputs) ** 2)

    params = update_parameters(model, train_loss,
        step_size=0.5, first_order=False)

    assert train_loss.item() == 6272.
    assert list(params.keys()) == ['weight']
    assert torch.all(params['weight'].data == torch.tensor([[-390., -613., -723.]]))

    """
    The new loss function (still with respect to the weights of the model w) is
    defined as
        g(w) = 17 * w'_1 + 19 * w'_2 + 23 * w'_3
             = 17 * (w_1 - 0.5 * df / dw_1) + 19 * (w_2 - 0.5 * df / dw_2) + 23 * (w_3 - 0.5 * df / dw_3)
             =   17 * (w_1 - 0.5 *  7 * (7 * w_1 + 11 * w_2 + 13 * w_3))
               + 19 * (w_2 - 0.5 * 11 * (7 * w_1 + 11 * w_2 + 13 * w_3))
               + 23 * (w_3 - 0.5 * 13 * (7 * w_1 + 11 * w_2 + 13 * w_3))
             =   (17 - 17 * 24.5 - 19 * 38.5 - 23 * 45.5) * w_1
               + (19 - 17 * 38.5 - 19 * 60.5 - 23 * 71.5) * w_2
               + (23 - 17 * 45.5 - 19 * 71.5 - 23 * 84.5) * w_3
             = -2177.5 * w_1 - 3429.5 * w_2 - 4052.5 * w_3

    Therefore the gradient of the function g with respect to w (and evaluated
    on w = [2, 3, 5]) is:
        dg / dw_1 = -2177.5
        dg / dw_2 = -3429.5
        dg / dw_3 = -4052.5
    """
    test_inputs = torch.tensor([[17., 19., 23.]])
    test_loss = model(test_inputs, params=params)

    grads = torch.autograd.grad(test_loss, model.parameters())

    assert test_loss.item() == -34906.
    assert len(grads) == 1
    assert torch.all(grads[0].data == torch.tensor([[-2177.5, -3429.5, -4052.5]]))

def test_update_parameters_first_order(model):
    """
    The loss function (with respect to the weights of the model w) is defined as
        f(w) = 0.5 * (29 * w_1 + 32 * w_2 + 37 * w_3) ** 2
    with w = [2, 3, 5].

    The gradient of the function f with respect to w, and evaluated
    on w = [2, 3, 5] is:
        df / dw_1 = 29 * (29 * w_1 + 31 * w_2 + 37 * w_3) =  9744
        df / dw_2 = 31 * (29 * w_1 + 31 * w_2 + 37 * w_3) = 10416
        df / dw_3 = 37 * (29 * w_1 + 31 * w_2 + 37 * w_3) = 12432

    The updated parameter w' is then given by one step of gradient descent,
    with step size 0.5:
        w'_1 = w_1 - 0.5 * df / dw_1 = 2 - 0.5 *  9744 = -4870
        w'_2 = w_2 - 0.5 * df / dw_2 = 3 - 0.5 * 10416 = -5205
        w'_3 = w_3 - 0.5 * df / dw_3 = 5 - 0.5 * 12432 = -6211
    """
    train_inputs = torch.tensor([[29., 31., 37.]])
    train_loss = 0.5 * (model(train_inputs) ** 2)

    params = update_parameters(model, train_loss,
        step_size=0.5, first_order=True)

    assert train_loss.item() == 56448.
    assert list(params.keys()) == ['weight']
    assert torch.all(params['weight'].data == torch.tensor([[-4870., -5205., -6211.]]))

    """
    The new loss function (still with respect to the weights of the model w) is
    defined as
        g(w) = 0.5 * (41 * w'_1 + 43 * w'_2 + 47 * w'_3) ** 2

    Since we computed w' with the first order approximation, the gradient of the
    function g with respect to w, and evaluated on w = [2, 3, 5], is:
        dg / dw_1 = 41 * (41 * w'_1 + 43 * w'_2 + 47 * w'_3) = -29331482
        dg / dw_2 = 43 * (41 * w'_1 + 43 * w'_2 + 47 * w'_3) = -30762286
        dg / dw_3 = 47 * (41 * w'_1 + 43 * w'_2 + 47 * w'_3) = -33623894
    """
    test_inputs = torch.tensor([[41., 43., 47.]])
    test_loss = 0.5 * (model(test_inputs, params=params) ** 2)

    grads = torch.autograd.grad(test_loss, model.parameters())

    assert test_loss.item() == 255900008448.
    assert len(grads) == 1
    assert torch.all(grads[0].data == torch.tensor([[-29331482., -30762286., -33623894.]]))
