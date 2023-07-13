"""
Classes for different GAN architectures
VanillaGAN : Goodfellow, Ouget, Mirza, Xu & Warde-Farley (2014) Generative Adversarial Nets
WassersteinGAN : Arjovsky, Chintala & Bottou (2017) Wasserstein GAN
FisherGAN : Mroueh & Sercu (2017) Fisher GAN
"""

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import imageio

from .helpers import get_device, one_hot_embedding
from .losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence, kl_divergence


class VanillaGAN():
    """
    Vanilla GAN and base class for different GAN architectures.

    Defines functions train(), _train_epoch() and sample_generator()

    Attributes:
        num_steps (int): The number of weight updates (critic). Generator
                         updates are num_steps/critic_iterations
    """

    def __init__(self, generator, critic, evidentialnn, g_optimizer, c_optimizer,
                 e_optimizer, critic_iterations=5, verbose=0, print_every=100,
                 use_cuda=False, training_gif=False):
        """
        Args:
            generator : pytorch nn module
              A pytorch module that outputs artificial data samples

            critic: pytorch nn module
              A pytorch module that discriminates between real and fake data

            g_optimizer, c_optimizer : pytorch optimizer object

            critic_iterations : int

            verbose : int

            print_every : int

            use_code : bool
        """
        self.use_cuda = use_cuda
        self.generator = generator
        self.g_opt = g_optimizer

        self.critic = critic
        self.c_opt = c_optimizer
        self.critic_iterations = critic_iterations

        self.evidentialnn = evidentialnn
        self.n_classes = evidentialnn.aux_input_size
        self.e_opt = e_optimizer

        # Monitoring
        self.num_steps = 0
        self.verbose = verbose
        self.print_every = print_every
        self.losses = {'generator_iter': [], 'generator': [],
                       'critic': [], 'discrimination': [], 'evidentialnn': []}
        self.maj_data = []
        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()
            self.evidentialnn.cuda()

        # Training gif
        self.training_gif = training_gif
        if training_gif:
            # Fix latents to see how image generation improves during training
            self.fixed_latents = Variable(self.generator.sample_latent(1000))
            if self.use_cuda:
                self.fixed_latents = self.fixed_latents.cuda()
            self.training_progress_images = []

    def _critic_train_iteration(self, data, aux_data):
        """ """
        # Get data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)
        ones = torch.ones(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()
            ones = ones.cuda()

        # Generate fake data
        generated_data, z = self.sample_generator(batch_size, aux_data)

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data).clamp(-0.0001, 0.0001)
        #print("OUTPUTS CRITIC: ",d_real)
        d_generated = self.critic(
            generated_data, aux_data).clamp(-0.0001, 0.0001)

        # Create total loss and optimize
        g_loss = (ones - d_generated).log().mean()
        d_distance = d_real.log().mean() + g_loss
        # We do gradient descent, not ascent as described in the original paper
        c_loss = -d_distance

        c_loss.backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Record loss
        if self.verbose > 1:
            if self.critic.training_iterations % self.print_every == 0:
                self.losses['critic'].append(-d_distance.data.numpy().item())
                self.losses["discrimination"].append(
                    d_generated.mean().data.numpy().item())
                self.losses['generator'].append(g_loss.data.numpy().item())
                self.losses['generator_iter'].append(
                    self.generator.training_iterations)

    def _evidentialnn_train_iteration(self, data, aux_data, epoch_num):
        # print("test")
        # Get data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)
        ones = torch.ones(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()
            ones = ones.cuda()

        # Generate fake data
        generated_data, z = self.sample_generator(batch_size, aux_data)

        # zero the parameter gradients
        self.e_opt.zero_grad()

        # forward
        # track history if only in train
        # y = one_hot_embedding(labels, num_classes) ###########
        #y = y.to(device)
        outputs = self.evidentialnn(data, aux_data)
        labels = torch.argmax(aux_data, dim=1)
        _, preds = torch.max(outputs, 1)
        #print("aux_data :",aux_data.float())
        #epoch_num =1
        loss = edl_log_loss(
            outputs, aux_data.float(), epoch_num, self.n_classes, 10
        )

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        u = self.n_classes / torch.sum(alpha, dim=1, keepdim=True)

        total_evidence = torch.sum(evidence, 1, keepdim=True)
        mean_evidence = torch.mean(total_evidence)
        mean_evidence_succ = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * match
        ) / torch.sum(match + 1e-20)
        mean_evidence_fail = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * (1 - match)
        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        # SECOND KL REGUL TERM
        #ev_output = self.evidentialnn(
        #    generated_data, aux_data)
        #print("ev_output of ",generated_data," for epoch num ",epoch_num," is ",ev_output)
        #print("ev_output: ", ev_output)
        #evidence = relu_evidence(ev_output)
        #alpha = evidence + 1
        #kl_alpha = (alpha - 1) * (1 - aux_data) + 1
        #S = torch.sum(alpha, dim=1, keepdim=True)
        #A = torch.sum(aux_data * (torch.log(S) -
        #              torch.log(alpha)), dim=1, keepdim=True)
        #kl_div2 = 1*kl_divergence(kl_alpha, self.n_classes)

        #loss = loss #+ kl_div2.mean()
        loss.backward()
        self.e_opt.step()

    def _pre_train_evidentialnn(self, data_loader,epoch_num):
        for i, data in enumerate(data_loader):
            self.num_steps += 1

            if len(data[1].shape) == 1:
                data[1] = data[1].unsqueeze(dim=1)
            #indices = torch.argmax(data[1], dim=1)
            #print("data pre reverse: ",data[1])
            #print("data post reverse: ",indices)

            self._evidentialnn_train_iteration(
                data[0], data[1], epoch_num=epoch_num)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data, z = self.sample_generator(batch_size, aux_data)

        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = -d_generated.mean()

        # Calculate KL divergence
        ev_output = self.evidentialnn(
            generated_data, aux_data)
        evidence = relu_evidence(ev_output)
        alpha = evidence + 1
        kl_alpha = (alpha - 1) * (1 - aux_data.float()) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(aux_data * (torch.log(S) -
                      torch.log(alpha)), dim=1, keepdim=True)
        u = 2 / torch.sum(alpha, dim=1, keepdim=True)
        kl_div = 5*kl_divergence(kl_alpha, self.n_classes)
        #print("g_loss: ", g_loss)
        #print("kl_div: ", kl_div)
        #print("generated_data: ",generated_data.mean(),", g_loss: ",g_loss) #," and u: ",u.mean())

        l2_strengh = 0.1
        l2_regul = l2_strengh* torch.square(u-1)
        z_strengh = 0.4
        l2_regul_z = z_strengh* torch.square(z-generated_data)
        #print("g_loss: ", g_loss)
        #print("kl_div: ", kl_div)
        #print("generated_data: ",generated_data.mean(),", g_loss: ",g_loss," and u: ",u.mean())
        g_loss = g_loss #+ l2_regul_z.mean() + l2_regul.mean()
        #g_loss = g_loss #+ l2_regul.mean()
        

        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1


    def _train_epoch(self, data_loader, epoch_num):
        for i, data in enumerate(data_loader):
            if self.generator.training_iterations < 10:
                critic_iterations = self.critic_iterations * 5
            else:
                critic_iterations = self.critic_iterations

            self.num_steps += 1

            if len(data[1].shape) == 1:
                data[1] = data[1].unsqueeze(dim=1)
            #indices = torch.argmax(data[1], dim=1)
            #print("data pre reverse: ",data[1])
            #print("data post reverse: ",indices)

            self._critic_train_iteration(data[0], data[1])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % critic_iterations == 0:
                self._generator_train_iteration(data[0], data[1])

            #self._evidentialnn_train_iteration(
            #    data[0], data[1], epoch_num=epoch_num)

            if self.verbose > 2:
                if i % self.print_every == 0:
                    print("Iteration {}".format(i + 1))
                    print("C: {}".format(self.losses['critic'][-1]))
                    if self.num_steps > critic_iterations:
                        print("G: {}".format(self.losses['generator'][-1]))

    def train(self, data_loader, epochs, maj_data):
        """
        Train the GAN generator and critic on the data given by the data_loader. GAN
        needs to be trained before synthetic data can be created.

        Arguments
        ---------
        data_loader :

        epochs : int
        Number of runs through the data (epochs)
        """
        self.maj_data = maj_data
        # PRE-TRAINING EVIDENTIAL NN
        
        if self.generator.training_iterations==0:
            #print("PRE-TRAINING EDL...")
            for epoch in range(500):
                self._pre_train_evidentialnn(data_loader=data_loader, epoch_num=epoch)


        compteur = 0
        #print("TRAINING GAN...num ", self.generator.training_iterations)
        for epoch in range(epochs):
            #print("\nEpoch {}".format(epoch + 1))
            compteur = compteur+1
            self._train_epoch(data_loader, epoch_num=compteur)

            if self.training_gif > 0:
                # Create plot
                sim_data = self.generator(self.fixed_latents).cpu().data
                no_vars = sim_data.shape[1]
                combinations = [(x, y) for x in range(no_vars)
                                for y in range(no_vars) if y > x]
                fig, axes = plt.subplots(nrows=no_vars-1, ncols=no_vars-1, squeeze=False,
                                         figsize=(10, 10))

                for i, j in combinations:
                    # , rasterized=True
                    axes[(i, j-1)].scatter(*sim_data[:, (i, j)].t())
                    axes[(i, j-1)].grid()

                fig.suptitle(
                    f"Generator iteration {self.generator.training_iterations}")
                # Add image grid to training progress
                # Keep limits constant
                #ax.set_ylim(0, y_max)
                # Used to return the plot as an image rray
                fig.canvas.draw()       # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                self.training_progress_images.append(image)
                plt.close()

    def sample_generator(self, num_samples, aux_data=None):
        """
        Generate num_samples observations from the generator.

        Arguments
        ---------
        num_samples: int
        Number of observations to generate

        aux_data: object
        Auxiliary data in the format used to train the generator
        """
        latent_samples = Variable(self.generator.sample_latent(num_samples))

        # generate a random permutation of indices
        random_indices = torch.randperm(num_samples)[:self.maj_data.shape[0]]

        # retrieve the corresponding rows
        random_rows = self.maj_data[random_indices]
        random_rows = Variable(torch.tensor(random_rows,dtype=torch.float32))
        #print("RANDOM MAJORITYS: ", random_rows)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        if self.generator.aux_dim == 0:
            generated_data = self.generator(random_rows)
        else:
            #print("shape selected rows: ",random_rows.shape)
            #print("aux_data")
            #print(aux_data)
            generated_data = self.generator(random_rows, aux_data)
            
        return generated_data, random_rows

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]

    def save_training_gif(self, path, step=1, loop=True):
        if self.training_gif > 0:
            training_progress_images = self.training_progress_images[::step]
            imageio.mimsave(path, training_progress_images,
                            subrectangles=True, loop=loop)
        else:
            ValueError(
                "Must set training_gif=True during training to plot a training gif")

    def plot_training(self):
        """
        Plot generator and critic loss and architecture-specific statistics
        over training iterations
        """
        fig = Figure()
        FigureCanvas(fig)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        losses = self.losses.copy()
        generator_iter = losses.pop("generator_iter")
        for key, data_list in losses.items():
            if key in ["generator", "critic"]:
                ax1.plot(generator_iter, data_list, label=key)
            elif key in ["distance"]:
                ax2.plot(generator_iter, data_list, label=key)
            else:
                ax3.plot(generator_iter, data_list, label=key)
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")

        return fig


class WassersteinGAN(VanillaGAN):
    """
    Class for Critic GAN with Wasserstein loss function
    """

    def __init__(self, generator, critic, evidentialnn, g_optimizer, c_optimizer, e_optimizer,
                 gp_weight=10, critic_iterations=5, verbose=0, print_every=100,
                 use_cuda=False, **kwargs):
        super().__init__(generator=generator, critic=critic, evidentialnn = evidentialnn,
                         g_optimizer=g_optimizer, c_optimizer=c_optimizer, e_optimizer = e_optimizer,
                         critic_iterations=critic_iterations,
                         verbose=verbose, print_every=print_every, use_cuda=use_cuda,
                         **kwargs)

        self.gp_weight = gp_weight

        # Monitoring
        self.losses = {'generator_iter': [],
                       'generator': [], 'critic': [], 'penalty': []}

        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()

    def _critic_train_iteration(self, data, aux_data):
        # Get generated data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()

        # Generate fake data
        generated_data, z = self.sample_generator(batch_size, aux_data)

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data)
        d_generated = self.critic(generated_data, aux_data)

        # Get gradient penalty
        gradient_penalty = self.gp_weight * \
            self._gradient_penalty(data, generated_data, aux_data)

        # Create total loss and optimize
        g_loss = d_generated.mean()
        d_distance = d_real.mean() - g_loss
        # The Wasserstein distance is the supremum (maximum) between the two expectations
        # in order to maximize, we minimize the negative loss and
        # add the penalty
        c_loss = -d_distance + gradient_penalty

        c_loss.backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Record loss
        if self.verbose > 1:
            if self.critic.training_iterations % self.print_every == 0:
                self.losses['penalty'].append(
                    gradient_penalty.data.numpy().item())
                self.losses['critic'].append(-d_distance.data.numpy().item())
                self.losses['generator'].append(-g_loss.data.numpy().item())
                self.losses['generator_iter'].append(
                    self.generator.training_iterations)
                
    def _evidentialnn_train_iteration(self, data, aux_data, epoch_num):
        # print("test")
        # Get data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)
        ones = torch.ones(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()
            ones = ones.cuda()

        # Generate fake data
        generated_data, z = self.sample_generator(batch_size, aux_data)

        # zero the parameter gradients
        self.e_opt.zero_grad()

        # forward
        # track history if only in train
        # y = one_hot_embedding(labels, num_classes) ###########
        #y = y.to(device)
        outputs = self.evidentialnn(data, aux_data)
        labels = torch.argmax(aux_data, dim=1)
        _, preds = torch.max(outputs, 1)
        #print("aux_data :",aux_data.float())
        #epoch_num =1
        loss = edl_mse_loss(
            outputs, aux_data.float(), epoch_num, 2, 10
        )

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        u = 2 / torch.sum(alpha, dim=1, keepdim=True)
        #print("u= ",u.mean())
        total_evidence = torch.sum(evidence, 1, keepdim=True)
        mean_evidence = torch.mean(total_evidence)
        mean_evidence_succ = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * match
        ) / torch.sum(match + 1e-20)
        mean_evidence_fail = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * (1 - match)
        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        # SECOND KL REGUL TERM
        #ev_output = self.evidentialnn(
        #    generated_data, aux_data)
        #print("ev_output of ",generated_data," for epoch num ",epoch_num," is ",ev_output)
        #print("ev_output: ", ev_output)
        #evidence = relu_evidence(ev_output)
        #alpha = evidence + 1
        #kl_alpha = (alpha - 1) * (1 - aux_data) + 1
        #S = torch.sum(alpha, dim=1, keepdim=True)
        #A = torch.sum(aux_data * (torch.log(S) -
        #              torch.log(alpha)), dim=1, keepdim=True)
        #kl_div2 = 1*kl_divergence(kl_alpha, self.n_classes)

        #loss = loss #+ kl_div2.mean()
        loss.backward()
        self.e_opt.step()

    def _pre_train_evidentialnn(self, data_loader,epoch_num):
        for i, data in enumerate(data_loader):
            self.num_steps += 1

            if len(data[1].shape) == 1:
                data[1] = data[1].unsqueeze(dim=1)
            #indices = torch.argmax(data[1], dim=1)
            #print("data pre reverse: ",data[1])
            #print("data post reverse: ",indices)

            self._evidentialnn_train_iteration(
                data[0], data[1], epoch_num=epoch_num)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data, z = self.sample_generator(batch_size, aux_data)

        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = -d_generated.mean()

        # Calculate KL divergence
        ev_output = self.evidentialnn(
            generated_data, aux_data)
        evidence = relu_evidence(ev_output)
        alpha = evidence + 1
        kl_alpha = (alpha - 1) * (1 - aux_data.float()) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(aux_data * (torch.log(S) -
                      torch.log(alpha)), dim=1, keepdim=True)
        u = 2 / torch.sum(alpha, dim=1, keepdim=True)
        kl_div = 5*kl_divergence(kl_alpha, self.n_classes)
        #print("g_loss: ", g_loss)
        #print("kl_div: ", kl_div)
        #print("generated_data: ",generated_data.mean(),", g_loss: ",g_loss) #," and u: ",u.mean())

        l2_strengh = 1
        l2_regul = l2_strengh* torch.square(u-1)
        z_strengh = 1
        l2_regul_z = z_strengh* torch.square(z-generated_data)
        #print("g_loss: ", g_loss)
        #print("kl_div: ", kl_div)
        #print("generated_data: ",generated_data.mean(),", g_loss: ",g_loss," and u: ",u.mean())
        g_loss = g_loss + l2_regul_z.mean() + l2_regul.mean()
        #g_loss = g_loss #+ l2_regul.mean()
        

        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1

    def _gradient_penalty(self, real_data, generated_data, aux_data):
        assert real_data.size() == generated_data.size(), ('real and generated mini batches must '
                                                           'have same size ({a} and {b})').format(
                                                               a=real_data.size(),
                                                               b=generated_data.size())
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, *[1 for _ in range(real_data.dim()-1)])
        #alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolated = alpha * real_data.data + \
            (1. - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate distance of interpolated examples
        d_interpolated = self.critic(interpolated, aux_x=aux_data)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   d_interpolated.size()).cuda()
                               if self.use_cuda else torch.ones(d_interpolated.size()),
                               create_graph=True, retain_graph=True,
                               only_inputs=True
                               )[0]

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        # if self.verbose > 0:
        #    if i % self.print_every == 0:
        #        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)
        return ((gradients_norm - 1) ** 2).mean()


class FisherGAN(VanillaGAN):
    """
    Class for GAN with Fisher loss function
    """

    def __init__(self, generator, critic, g_optimizer, c_optimizer,
                 critic_iterations=3, penalty=1e-6, kl_div_strength=0.4, z_strength=0.4, verbose=0, print_every=100,
                 use_cuda=False, **kwargs):
        super().__init__(generator=generator, critic=critic,
                         g_optimizer=g_optimizer, c_optimizer=c_optimizer,
                         critic_iterations=critic_iterations,
                         verbose=verbose, print_every=print_every,
                         use_cuda=use_cuda, **kwargs)

        # Monitoring
        self.losses = {'generator_iter': [], 'generator': [], 'critic': [], "distance": [],
                       'lagrange_multiplier': [], "constraint": []}

        self.penalty = penalty
        self.lagrange_mult = torch.FloatTensor([0])
        self.lagrange_mult = Variable(self.lagrange_mult, requires_grad=True)

        self.kl_div_strength = kl_div_strength
        self.z_strength = z_strength

        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()
            self.lagrange_mult.cuda()
            self.penalty.cuda()

    def _critic_train_iteration(self, data, aux_data):
        # Get generated data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)
        one = torch.ones(1)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()
            one = one.cuda()

        # Generate fake data
        generated_data, z = self.sample_generator(batch_size, aux_data)

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data)
        d_generated = self.critic(generated_data, aux_data)
        # Calculate loss as different in means of critic output
        g_loss = d_generated.mean()
        distance = d_real.mean() - g_loss

        # Calculate constraint \Omega
        constraint = (0.5*(d_real**2).mean() + 0.5*(d_generated**2).mean())

        # Create total loss and optimize
        c_loss = (distance + self.lagrange_mult * (one-constraint)
                  - self.penalty/2 * (one-constraint)**2)

        # Maximize critic weights w.r.t augmented lagrangian L by minimizing negative L
        (-c_loss).backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Minimize lagrange multiplier lambda w.r.t to loss and quadratic penalty rho
        self.lagrange_mult.data += self.penalty * self.lagrange_mult.grad.data
        self.lagrange_mult.grad.data.zero_()

        # Record loss
        if self.verbose > 1:
            if self.critic.training_iterations % self.print_every == 0:
                self.losses['lagrange_multiplier'].append(
                    self.lagrange_mult.data.numpy().item())
                self.losses['constraint'].append(
                    constraint.data.numpy().item())
                self.losses["distance"].append(distance.data.numpy().item())
                self.losses['critic'].append(-c_loss.data.numpy().item())
                self.losses['generator'].append(-g_loss.data.numpy().item())
                self.losses['generator_iter'].append(
                    self.generator.training_iterations)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data, z = self.sample_generator(batch_size, aux_data)
        #
        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = -d_generated.mean()

        # Calculate KL divergence
        ev_output = self.evidentialnn(
            generated_data, aux_data)
        evidence = relu_evidence(ev_output)
        alpha = evidence + 1
        kl_alpha = (alpha - 1) * (1 - aux_data.float()) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(aux_data * (torch.log(S) -
                      torch.log(alpha)), dim=1, keepdim=True)
        u = 2 / torch.sum(alpha, dim=1, keepdim=True)

        #kl_div_strength = self.kl_div_strength
        kl_div = self.kl_div_strength*kl_divergence(kl_alpha, self.n_classes)
        #l2_strength = 0.1 #self.l2_strengh
        l2_regul = self.kl_div_strength* torch.square(u-1)
        #z_strength = self.z_strength
        l2_regul_z = self.z_strength* torch.square(z-generated_data)
        #print("g_loss: ", g_loss)
        #print("kl_div: ", kl_div)
        #print("generated_data: ",generated_data.mean(),", g_loss: ",g_loss," and u: ",u.mean())
        g_loss = g_loss + l2_regul_z.mean() + l2_regul.mean()

        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1
