from torch import nn
from torch.nn import init


class Deep4Net(nn.Sequential):
    def __init__(
            self,
            in_chans,
            n_classes,
            input_window_samples,
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=10,
            n_filters_3=100,
            filter_length_3=10,
            n_filters_4=200,
            filter_length_4=10,
            drop_prob=0.5,
            double_time_convs=False,
            batch_norm=True,
            batch_norm_alpha=0.1,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.conv_stride = 1
        self.pool_stride = self.pool_time_stride
        self.n_filters_conv = self.n_filters_spat

        self.first_block = nn.Sequential(
            nn.Conv2d(
                1,
                self.n_filters_time,
                (1, self.filter_time_length),
                stride=(1, 1),
            ),
            nn.Conv2d(
                self.n_filters_time,
                self.n_filters_spat,
                (self.in_chans, 1),
                stride=(self.conv_stride, 1),
                bias=not self.batch_norm,
            ),

            nn.BatchNorm2d(
                self.n_filters_conv,
                momentum=self.batch_norm_alpha,
                affine=True,
                eps=1e-5,
            ),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, self.pool_time_length), stride=(1, self.pool_stride)
            ),
        )

        self.second_block = self.add_conv_pool_block(self.n_filters_conv, self.n_filters_2, self.filter_length_2)
        self.third_block = self.add_conv_pool_block(
            self.n_filters_2, self.n_filters_3, self.filter_length_3
        )
        self.fourth_block = self.add_conv_pool_block(
            self.n_filters_3, self.n_filters_4, self.filter_length_4
        )

        self.conv_classifier = nn.Sequential(
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (1, 7),
                bias=True,
            ),
        )

        self.softmax = nn.Sequential(nn.LogSoftmax(dim=1))

        param_dict_first_block = dict(list(self.first_block.named_parameters()))
        param_dict_second_block = dict(list(self.second_block.named_parameters()))
        param_dict_third_block = dict(list(self.third_block.named_parameters()))
        param_dict_fourth_block = dict(list(self.fourth_block.named_parameters()))
        param_dict_conv_classifier = dict(list(self.conv_classifier.named_parameters()))

        init.xavier_uniform_(param_dict_first_block['0.weight'], gain=1)
        init.constant_(param_dict_first_block['0.bias'], 0)
        init.xavier_uniform_(param_dict_first_block['1.weight'], gain=1)
        init.constant_(param_dict_first_block['2.weight'], 1)
        init.constant_(param_dict_first_block['2.bias'], 0)
        init.xavier_uniform_(param_dict_second_block["1.weight"], gain=1)
        init.xavier_uniform_(param_dict_third_block["1.weight"], gain=1)
        init.xavier_uniform_(param_dict_fourth_block["1.weight"], gain=1)
        init.constant_(param_dict_second_block["2.weight"], 1)
        init.constant_(param_dict_second_block["2.bias"], 0)
        init.constant_(param_dict_third_block["2.weight"], 1)
        init.constant_(param_dict_third_block["2.bias"], 0)
        init.constant_(param_dict_fourth_block["2.weight"], 1)
        init.constant_(param_dict_fourth_block["2.bias"], 0)
        init.xavier_uniform_(param_dict_conv_classifier['0.weight'], gain=1)
        init.constant_(param_dict_conv_classifier['0.bias'], 0)

    def add_conv_pool_block(
            self, n_filters_before, n_filters, filter_length
    ):
        default_block = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(
                n_filters_before,
                n_filters,
                (1, filter_length),
                stride=(self.conv_stride, 1),
                bias=not self.batch_norm,
            ),
            nn.BatchNorm2d(
                n_filters,
                momentum=self.batch_norm_alpha,
                affine=True,
                eps=1e-5,
            ),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, self.pool_time_length),
                stride=(1, self.pool_stride),
            ),
        )

        return default_block

    def forward(self, input):
        output = self.first_block(input)
        output = self.second_block(output)
        output = self.third_block(output)
        output = self.fourth_block(output)
        output = self.conv_classifier(output)
        output = output.view(output.size(0), -1)
        output = self.softmax(output)

        return output
